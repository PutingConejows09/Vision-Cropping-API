import os
print("GOOGLE_APPLICATION_CREDENTIALS =", os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))

# Optional: confirm Vision client can init
try:
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    print("Vision client initialized OK.")
except Exception as e:
    print("Vision init error:", e)
