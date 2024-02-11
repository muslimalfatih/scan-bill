# Scan Bill 

This is a Python application designed to process images of receipts or invoices using the Donut model for Optical Character Recognition (OCR) and Natural Language Processing (NLP). The application extracts text data from receipt images and provides structured information such as menu items, subtotal, and total.

## Demo
https://cln.sh/04Q6gqrT

## Dependencies

- Python 3.x
- FastAPI
- PyTorch
- Transformers
- PIL (Python Imaging Library)

## Setup
Clone this repository to your local machine:
```
git clone https://github.com/muslimalfatih/scan-bill
```

Install dependencies
```
pip install -r requirements.txt
```

## Usage
Start the FastAPI server:
```
uvicorn main:app --reload
```
- Open your web browser and navigate to http://localhost:8000 to access the application.
- Upload an image of a receipt or invoice to the app. You can try sample receipts in `static` folder
- Click the **Process Receipt** button to extract information from the uploaded image.
- View the extracted data on the results page.

## About the Donut Model
The Donut model is a state-of-the-art deep learning model developed by Naver AI Lab for document understanding tasks. It combines computer vision and natural language processing techniques to extract structured information from unstructured documents such as receipts, invoices, and forms. The model is fine-tuned for receipt processing tasks and achieves high accuracy in recognizing text and extracting relevant information.

## References
- FastAPI Documentation: https://fastapi.tiangolo.com/
- Transformers Documentation: https://huggingface.co/transformers/
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- PIL Documentation: https://pillow.readthedocs.io/en/stable/

**Note**: Ensure that you have a stable internet connection when running the application for the first time to download the Donut model and processor. Subsequent runs will use the downloaded model files stored locally.