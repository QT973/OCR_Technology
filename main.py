import streamlit as st
from paddleocr import PaddleOCR
import cv2
import numpy as np
from PIL import Image

class OCRApp:
    def __init__(self, ocr_model):
        self.ocr_model = ocr_model
        self.test = None

    def process_frame(self, frame, input_text):
        # Chuyển frame từ BGR sang RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.ocr_model.ocr(np.asarray(frame_rgb), cls=True)

        found_text = False  # Biến kiểm tra xem có tìm thấy văn bản hay không
        text = None
        if result[0]:  # Nếu có kết quả OCR
            for line in result[0]:
                box = [(int(x), int(y)) for x, y in line[0]]
                text = line[1][0]
                found_text = True  # Đánh dấu tìm thấy văn bản

                if text == input_text:
                    color = (0, 255, 0)
                    self.test = "Pass"
                else:
                    color = (0, 0, 255)
                    self.test = "No pass"

                # Vẽ khung và văn bản lên frame
                for i in range(4):
                    cv2.line(frame, box[i], box[(i + 1) % 4], color, 2)
                cv2.putText(frame, text, (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if not found_text:
            self.test = "No text detected"

        return frame, self.test,text

def main():
    # Khởi tạo PaddleOCR và ứng dụng OCRApp
    ocr_app = OCRApp(PaddleOCR(use_angle_cls=True, lang='vi'))

    # Giao diện Streamlit
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>OCR Technology</h1>", unsafe_allow_html=True)
    st.sidebar.image("logo_truong.png", use_container_width=True)
    
    # Chọn nguồn đầu vào
    input_source = st.radio("Chọn nguồn video:", ("Webcam", "RTSP/Video URL", "Upload Image"))
    

    if input_source == "Upload Image":
        uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        text_test = st.text_input("Input text for OCR:")
        if uploaded_file is not None:
            col1, col2, col3 = st.columns([5,5,3 ])
            # Hiển thị ảnh gốc
            original_image = Image.open(uploaded_file)
            frame = np.array(original_image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            processed_image, test,text = ocr_app.process_frame(frame, text_test)
            with col1:
                st.image(original_image, caption="Input image", use_container_width=True)
                bt = st.button("Process")
             # Nút xử lý
            if bt:
                with col2:
                    st.image(
                        cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB),
                        caption="Processed image",
                        use_container_width=True
                    )               
                with col3:
                    st.caption(f"Results text: :blue[{text}]")
                    if test == "Pass":
                        st.header(f"Results: :green[{test}]")
                    else:
                        st.header(f"Results: :red[{test}]")
        else:
            st.warning("Please upload an image.")

    elif input_source == "RTSP/Video URL":
        rtsp_url = st.text_input("Nhập URL video hoặc RTSP:")
        text_test = st.text_input("Input text for OCR:")
        if st.checkbox("Start Stream"):
            if not rtsp_url:
                st.error("Please provide a valid RTSP/Video URL.")
                return
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                st.error("Error: Could not open video stream.")
                return
            
            col1, col2, col3 = st.columns([5, 5, 3])
            with col1:
                st.markdown("<h3 style='text-align: center; color: #2196F3;'>Video Gốc</h3>", unsafe_allow_html=True)
                stream = st.empty()
            with col2:
                st.markdown("<h3 style='text-align: center; color: #2196F3;'>Video Qua Detect</h3>", unsafe_allow_html=True)
                stframe = st.empty()
            with col3:
                st.markdown("<h3 style='text-align: center; color: #2196F3;'>Kết Quả</h3>", unsafe_allow_html=True)
                result_text = st.empty()
                result_status = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.warning("Warning: Unable to read from video stream.")
                    break

                # Hiển thị video gốc
                stream.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                # Xử lý OCR và hiển thị video đã qua xử lý
                processed_frame, status, text = ocr_app.process_frame(frame, text_test)
                stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                
                # Cập nhật kết quả OCR trong cột col3
                result_text.caption(f"Results text: :blue[{text}]")
                if status == "Pass":
                    result_status.header(f"Results: :green[{status}]")
                else:
                    result_status.header(f"Results: :red[{status}]")

            cap.release()
            cv2.destroyAllWindows()

    elif input_source == "Webcam":
        text_test = st.text_input("Input text for OCR:")
        if st.checkbox("Start Webcam"):
            cap = cv2.VideoCapture(2)
            if not cap.isOpened():
                st.error("Error: Could not access webcam.")
                return
            col1, col2, col3 = st.columns([5, 5, 3])
            with col1:
                st.markdown("<h3 style='text-align: center; color: #2196F3;'>Video Gốc</h3>", unsafe_allow_html=True)
                stream = st.empty()
            with col2:
                st.markdown("<h3 style='text-align: center; color: #2196F3;'>Video Qua Detect</h3>", unsafe_allow_html=True)
                stframe = st.empty()
            with col3:
                st.markdown("<h3 style='text-align: center; color: #2196F3;'>Kết Quả</h3>", unsafe_allow_html=True)
                result_text = st.empty()
                result_status = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.warning("Warning: Unable to read from video stream.")
                    break

                # Hiển thị video gốc
                stream.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                # Xử lý OCR và hiển thị video đã qua xử lý
                processed_frame, status, text = ocr_app.process_frame(frame, text_test)
                stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                
                # Cập nhật kết quả OCR trong cột col3
                result_text.caption(f"Results text: :blue[{text}]")
                if status == "Pass":
                    result_status.header(f"Results: :green[{status}]")
                else:
                    result_status.header(f"Results: :red[{status}]")

            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
