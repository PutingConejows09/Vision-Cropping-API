# crop_receipt_run.py
# Simplified and guaranteed visible version.

import os
from google.cloud import vision
from PIL import Image

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def crop_image(image_path, box, output_path):
    with Image.open(image_path) as im:
        w, h = im.size
        left, top, right, bottom = box
        left   = clamp(left - 10, 0, w)
        top    = clamp(top - 10, 0, h)
        right  = clamp(right + 10, 0, w)
        bottom = clamp(bottom + 10, 0, h)
        cropped = im.crop((left, top, right, bottom))
        cropped.save(output_path)
        print(f"✅ Cropped saved: {output_path}")

def detect_and_crop(image_path):
    print(f"➡️ Processing {image_path} ...")
    client = vision.ImageAnnotatorClient()

    with open(image_path, "rb") as f:
        content = f.read()

    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)

    pages = response.full_text_annotation.pages
    if not pages:
        print(f"⚠️ No text found in {image_path}, skipping.")
        return

    all_x, all_y = [], []
    for page in pages:
        for block in page.blocks:
            for v in block.bounding_box.vertices:
                all_x.append(v.x)
                all_y.append(v.y)

    left, top, right, bottom = min(all_x), min(all_y), max(all_x), max(all_y)
    output_path = f"Cropped-{os.path.basename(image_path)}"
    crop_image(image_path, (left, top, right, bottom), output_path)

def main():
    # find all jpg and png in current folder
    images = [f for f in os.listdir() if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not images:
        print("⚠️ No image files (.jpg/.png) found in this folder.")
        return

    print(f"\n🔍 Found {len(images)} image file(s). Starting...\n")
    for img in images:
        try:
            detect_and_crop(img)
        except Exception as e:
            print(f"❌ Error with {img}: {e}")

    print("\n🎉 All done! Check your folder for files starting with 'Cropped-'.\n")

if __name__ == "__main__":
    main()
