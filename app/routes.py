from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import re
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

router = APIRouter()
templates = Jinja2Templates(directory="templates") 
router.mount("/static", StaticFiles(directory="static"), name="static")

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
task_prompt = "<s_cord-v2>"

def doc_to_text(input_img):
    # set model device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    try:
        # document preprocessing
        pixel_values = processor(input_img, return_tensors="pt").pixel_values
        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # sequence generation
        outputs = model.generate(
            pixel_values.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            max_length=model.decoder.config.max_position_embeddings,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            
            # modify parameters
            early_stopping=True,
            num_beams=2,
            output_scores=True,
        )
        
        # document post-processing: sequence token cleaning
        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
        print(sequence)

        # output conversion: token to json
        output = processor.token2json(sequence)

        return output

    except Exception as e:
        print(f"Error in doc_to_text: {e}")
        raise

@router.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/process_receipt", response_class=HTMLResponse)
async def process_receipt(request: Request, file: UploadFile = File(...)):
    try:
        contents = await file.read()  # Read the file content

        # Open the image using PIL
        input_img = Image.open(io.BytesIO(contents))

        # Process the receipt using the Donut model
        result = doc_to_text(input_img)

        # Check if required attributes are present in the result
        if 'menu' in result and 'sub_total' in result and 'total' in result:
            return templates.TemplateResponse("index.html", {"request": request, "output": result})
        else:
            return templates.TemplateResponse("index.html", {"request": request, "error": "Invalid receipt format"})

    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": f"Error processing receipt: {str(e)}"})
