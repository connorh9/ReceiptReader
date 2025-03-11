import openai
import cv2
import pytesseract
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv()
#api_key = os.getenv("openai_key")

client = openai.OpenAI(
    api_key=os.getenv("openai_key"),
)

def alterImage(img):
    #print(pytesseract)
    img = cv2.imread(img)
    resize_img = cv2.resize(img, (1800, 1800), interpolation=cv2.INTER_AREA)
    
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrasted_img = cv2.addWeighted(grayscale_img, 2, grayscale_img, -1, 0)
    reduced_noise = cv2.fastNlMeansDenoising(contrasted_img,None,30,7,21)

    ret, thresh_img = cv2.threshold(reduced_noise, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilation = cv2.dilate(thresh_img, rect_kernel, iterations = 1)

    text = pytesseract.image_to_string(dilation)
    return text

def getFormattedJson(text):
    prompt = """You are a receipt parser AI. I will give you raw text extracted from an image of a store receipt. I want you to return a JSON object of that text and
    include a field named expense_type where you infer what type of expense it is and categorize it based on the receipt. The JSON stucture should be: 
    {"total", "business", "items": [{"title", "quantity", "price"}], "timestamp", "expense_type}. The prices should be integers representing the number of pennies,
    ($1 = 100). Only return the JSON object to me. Here is the extracted text from the receipt: """ + text

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role":"user",
                "content": prompt
            }
        ],
        response_format={"type":"json_object"}
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    path = input("Enter image path: ")
    raw_text = alterImage(path)
    json_data = getFormattedJson(raw_text)
    print(json_data)