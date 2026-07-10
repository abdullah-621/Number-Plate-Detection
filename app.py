"""
Streamlit App: License Plate Detection + Tracking + OCR
---------------------------------------------------------
Pipeline: YOLO Detection -> ByteTrack -> Crop Plate -> OCR -> Print Once -> CSV

Run:
    pip install streamlit ultralytics easyocr opencv-python-headless
    streamlit run app.py
"""

import os
import csv
import tempfile

import cv2
import streamlit as st
import easyocr
from ultralytics import YOLO


# ---------- Page Config ----------
st.set_page_config(page_title="License Plate Detector", page_icon="🚗", layout="centered")
st.title("🚗 License Plate Detection & Recognition")
st.write("Upload a video, then press the **Detect** button. "
         "A CSV file containing the license plate numbers of all tracked vehicles will be generated.")

# ---------- Sidebar: model path ----------
st.sidebar.header("⚙️ Settings")
model_path = st.sidebar.text_input(
    "YOLO model (.pt) path",
    value="best.pt",
    help="best.pt"
)

if not os.path.isfile(model_path):
    st.error(
        f"❌ Model is not found: `{model_path}`\n\n"
        "Insure that:\n"
        "- The file is indeed at this path.\n"
        "- Or, set the correct path from the sidebar.\n"
        "- You can provide the full path including the `.pt` extension (e.g., `C:\\Users\\...\\best.pt`)."
    )
    st.stop()


# ---------- Cache the models so they load only once ----------
@st.cache_resource
def load_models(model_path):
    yolo_model = YOLO(model_path)
    ocr_reader = easyocr.Reader(['en'])
    return yolo_model, ocr_reader


# ---------- Core pipeline function ----------
def run_pipeline(video_path, model, reader, progress_bar, status_text):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    # avc1 (H.264) browser-এ play হয়, mp4v অনেক সময় হয় না — আগে avc1 try করছি
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))

    csv_path = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["ID", "Plate Number"])

    printed_ids = set()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if track_id in printed_ids:
                    continue

                plate_crop = frame[max(0, y1):y2, max(0, x1):x2]
                if plate_crop.size == 0:
                    continue

                ocr_result = reader.readtext(plate_crop)
                if ocr_result:
                    best = max(ocr_result, key=lambda r: r[2])
                    plate_text = best[1].strip()
                    if plate_text:
                        printed_ids.add(track_id)
                        csv_writer.writerow([track_id, plate_text])
                        csv_file.flush()
                        status_text.write(f"✅ ID {track_id}: **{plate_text}**")

        out.write(frame)

        if frame_count % 5 == 0 or frame_count == total_frames:
            progress_bar.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    out.release()
    csv_file.close()

    return out_video_path, csv_path, len(printed_ids)


# ---------- UI ----------
uploaded_video = st.file_uploader("Upload the video.", type=["mp4", "avi", "mov", "mkv"])

# session_state-এ আগের ফলাফল থাকলে সেটা মনে রাখা হয়, নাহলে download বাটনে ক্লিক করলে
# পুরো পেজ rerun হয়ে সব ফলাফল হারিয়ে যায়
if "output_video_path" not in st.session_state:
    st.session_state.output_video_path = None
    st.session_state.csv_path = None
    st.session_state.plate_count = None

if uploaded_video is not None:
    st.video(uploaded_video)

    if st.button("🔍 Detect", type="primary"):
        with st.spinner("Loading model...."):
            model, reader = load_models(model_path)

        # আপলোড করা ভিডিও temp ফাইলে সেভ করা
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.close()   # Windows-এ ফাইল lock এড়ানোর জন্য handle টা বন্ধ করে দিচ্ছি
        input_path = tfile.name

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        with st.spinner("Detection + Tracking + OCR in progress... it will take a little time."):
            output_video_path, csv_path, plate_count = run_pipeline(
                input_path, model, reader, progress_bar, status_text
            )

        # ফলাফল session_state-এ সেভ করা হচ্ছে যাতে পরে download বাটনে ক্লিক করলে হারিয়ে না যায়
        st.session_state.output_video_path = output_video_path
        st.session_state.csv_path = csv_path
        st.session_state.plate_count = plate_count

        # cleanup input temp file
        try:
            os.remove(input_path)
        except PermissionError:
            pass   # Windows মাঝেমধ্যে সাথে সাথে unlock করে না, সমস্যা নেই — temp ফোল্ডার নিজে থেকেই পরে পরিষ্কার হয়

# ---------- Results (session_state থেকে দেখানো হচ্ছে, তাই download-এর পরও থেকে যাবে) ----------
if st.session_state.csv_path is not None:
    st.success(f"✅ Done! A total of {st.session_state.plate_count} plates have been detected.")

    st.subheader("📄 Annotated Video")
    try:
        st.video(st.session_state.output_video_path)
    except Exception as e:
        st.warning(
            "⚠️ The video preview isn't showing here (it might be a browser codec issue), "
                "but the file has been created—you can download it below and watch it in VLC or any "
                f"player.\n\nDetails: {e}"
        )

    with open(st.session_state.output_video_path, "rb") as vf:
        st.download_button(
            label="Download Annotated Video",
            data=vf,
            file_name="output_annotated.mp4",
            mime="video/mp4",
            key="download_video",
        )

    st.subheader("⬇️ Download CSV")
    with open(st.session_state.csv_path, "rb") as f:
        st.download_button(
            label="Download CSV",
            data=f,
            file_name="plates_output.csv",
            mime="text/csv",
            key="download_csv",
        )