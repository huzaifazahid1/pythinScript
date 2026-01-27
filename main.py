# import os
# import torch
# from PIL import Image
# from tqdm import tqdm
# from transformers import (
#     AutoModel,
#     AutoProcessor,
#     InstructBlipProcessor,
#     InstructBlipForConditionalGeneration
# )

# # ==========================================
# # 📂 PATH SETUP
# # ==========================================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# IMAGE_DIR = os.path.join(BASE_DIR, "images")
# OUTPUT_HTML = os.path.join(BASE_DIR, "metadata_results.html")

# if not os.path.exists(IMAGE_DIR):
#     print("❌ 'images' folder not found next to main.py")
#     exit()

# # ==========================================
# # 🧠 TAXONOMY
# # ==========================================
# TAXONOMY = {
#     "Backgrounds": ["Abstract Pattern", "Texture", "Gradient Colors", "Bokeh Blur", "Geometric Shapes"],
#     "Business": ["Office Meeting", "Financial Charts", "Handshake", "Working on Laptop", "Corporate Team"],
#     "People": ["Portrait", "Crowd", "Family", "Fitness & Sports", "Happy Emotion"],
#     "Technology": ["Artificial Intelligence", "Coding & Programming", "Virtual Reality", "Circuit Board", "Smart Devices"],
#     "Travel": ["Mountains & Hiking", "Beach & Ocean", "Cityscape & Architecture", "Airport & Luggage", "Historical Monuments"],
#     "Nature": ["Forest & Trees", "Sky & Clouds", "Flowers & Garden", "Desert & Sand", "Waterfalls & Rivers"],
#     "Medical": ["Doctor & Nurse", "Hospital Bed", "Pills & Medicine", "Microscope & Lab", "Surgery"],
#     "Food": ["Fresh Fruits", "Fast Food", "Desserts & Cakes", "Coffee & Drinks", "Healthy Salad"],
#     "Sci-Fi & Fantasy": ["Space & Galaxy", "Cyberpunk & Futuristic", "Surrealism", "3D Render", "Aliens & Robots"],
#     "Animals & Wildlife": ["Pets", "Wild Animals", "Birds", "Marine Life", "Insects", "Farm Animals"]
# }

# # ==========================================
# # 🚀 LOAD MODELS
# # ==========================================
# def load_models():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"🖥️ Device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

#     print("🔹 Loading SigLIP...")
#     siglip_proc = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
#     siglip_model = AutoModel.from_pretrained(
#         "google/siglip-so400m-patch14-384"
#     ).to(device)

#     print("🔹 Loading InstructBLIP...")
#     blip_proc = InstructBlipProcessor.from_pretrained(
#         "Salesforce/instructblip-flan-t5-xl"
#     )
#     blip_model = InstructBlipForConditionalGeneration.from_pretrained(
#         "Salesforce/instructblip-flan-t5-xl",
#         torch_dtype=torch.float16,
#         device_map="auto"
#     )

#     print("✅ Models loaded\n")
#     return device, siglip_proc, siglip_model, blip_proc, blip_model

# # ==========================================
# # 🧠 IMAGE PROCESSING
# # ==========================================
# def process_image(img_path, device, s_proc, s_model, b_proc, b_model):
#     try:
#         image = Image.open(img_path).convert("RGB")
#     except:
#         return None

#     # ---- CATEGORY (SigLIP) ----
#     main_keys = list(TAXONOMY.keys())
#     inputs = s_proc(text=main_keys, images=image, return_tensors="pt", padding=True).to(device)

#     with torch.no_grad():
#         out = s_model(**inputs)

#     main_cat = main_keys[torch.sigmoid(out.logits_per_image).argmax().item()]

#     sub_keys = TAXONOMY[main_cat]
#     inputs_sub = s_proc(text=sub_keys, images=image, return_tensors="pt", padding=True).to(device)

#     with torch.no_grad():
#         out_sub = s_model(**inputs_sub)

#     sub_cat = sub_keys[torch.sigmoid(out_sub.logits_per_image).argmax().item()]

#     # ---- DESCRIPTION (InstructBLIP) ----
#     prompt = "Describe the image in detail including objects, style, lighting and mood."
#     inputs_blip = b_proc(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

#     with torch.no_grad():
#         gen = b_model.generate(**inputs_blip, max_new_tokens=90)

#     caption = b_proc.batch_decode(gen, skip_special_tokens=True)[0]

#     keywords = ", ".join(sorted(set([
#         w.strip(".,").lower()
#         for w in caption.split()
#         if len(w) > 4
#     ])))

#     return {
#         "filename": os.path.basename(img_path),
#         "image_path": f"images/{os.path.basename(img_path)}",
#         "main_cat": main_cat,
#         "sub_cat": sub_cat,
#         "caption": caption,
#         "keywords": keywords
#     }

# # ==========================================
# # 🧾 HTML GENERATOR
# # ==========================================
# def generate_html(results):
#     rows = ""
#     for r in results:
#         rows += f"""
#         <tr>
#             <td><img src="{r['image_path']}" width="120"></td>
#             <td>{r['main_cat']}</td>
#             <td>{r['sub_cat']}</td>
#             <td>{r['caption']}</td>
#             <td>{r['keywords']}</td>
#         </tr>
#         """

#     html = f"""
#     <!DOCTYPE html>
#     <html>
#     <head>
#         <meta charset="UTF-8">
#         <title>Image Metadata Results</title>
#         <style>
#             body {{ font-family: Arial, sans-serif; background:#f5f5f5; }}
#             table {{ border-collapse: collapse; width: 100%; background:white; }}
#             th, td {{ border: 1px solid #ddd; padding: 10px; vertical-align: top; }}
#             th {{ background: #222; color: white; }}
#             img {{ border-radius: 6px; }}
#         </style>
#     </head>
#     <body>
#         <h2>📸 Image Metadata Results</h2>
#         <table>
#             <tr>
#                 <th>Image</th>
#                 <th>Main Category</th>
#                 <th>Sub Category</th>
#                 <th>Description</th>
#                 <th>Keywords</th>
#             </tr>
#             {rows}
#         </table>
#     </body>
#     </html>
#     """

#     with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
#         f.write(html)

# # ==========================================
# # ▶️ MAIN
# # ==========================================
# if __name__ == "__main__":
#     device, s_proc, s_model, b_proc, b_model = load_models()

#     images = [
#         os.path.join(IMAGE_DIR, f)
#         for f in os.listdir(IMAGE_DIR)
#         if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
#     ]

#     print(f"📸 Found {len(images)} images\n")

#     results = []
#     for img in tqdm(images, desc="Processing Images"):
#         data = process_image(img, device, s_proc, s_model, b_proc, b_model)
#         if data:
#             results.append(data)

#     if results:
#         generate_html(results)
#         print(f"\n✅ DONE! HTML file created:\n{OUTPUT_HTML}")
#     else:
#         print("⚠️ No results generated.")
































































# import os
# import io
# import torch
# from PIL import Image
# from tqdm import tqdm
# from transformers import (
#     AutoModel,
#     AutoProcessor,
#     InstructBlipProcessor,
#     InstructBlipForConditionalGeneration
# )

# # ==========================================
# # 📂 PATH SETUP
# # ==========================================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# IMAGE_DIR = os.path.join(BASE_DIR, "images")
# OUTPUT_HTML = os.path.join(BASE_DIR, "metadata_results.html")
# COMPRESSED_DIR = os.path.join(BASE_DIR, "compressed")

# os.makedirs(COMPRESSED_DIR, exist_ok=True)

# if not os.path.exists(IMAGE_DIR):
#     print("❌ 'images' folder not found next to main.py")
#     exit()

# # ==========================================
# # 🧠 TAXONOMY
# # ==========================================
# TAXONOMY = {
#     "Backgrounds": ["Abstract Pattern", "Texture", "Gradient Colors", "Bokeh Blur", "Geometric Shapes"],
#     "Business": ["Office Meeting", "Financial Charts", "Handshake", "Working on Laptop", "Corporate Team"],
#     "People": ["Portrait", "Crowd", "Family", "Fitness & Sports", "Happy Emotion"],
#     "Technology": ["Artificial Intelligence", "Coding & Programming", "Virtual Reality", "Circuit Board", "Smart Devices"],
#     "Travel": ["Mountains & Hiking", "Beach & Ocean", "Cityscape & Architecture", "Airport & Luggage", "Historical Monuments"],
#     "Nature": ["Forest & Trees", "Sky & Clouds", "Flowers & Garden", "Desert & Sand", "Waterfalls & Rivers"],
#     "Medical": ["Doctor & Nurse", "Hospital Bed", "Pills & Medicine", "Microscope & Lab", "Surgery"],
#     "Food": ["Fresh Fruits", "Fast Food", "Desserts & Cakes", "Coffee & Drinks", "Healthy Salad"],
#     "Sci-Fi & Fantasy": ["Space & Galaxy", "Cyberpunk & Futuristic", "Surrealism", "3D Render", "Aliens & Robots"],
#     "Animals & Wildlife": ["Pets", "Wild Animals", "Birds", "Marine Life", "Insects", "Farm Animals"]
# }

# # ==========================================
# # 🔧 IMAGE COMPRESSION FUNCTION (~1MB)
# # ==========================================
# def compress_image_to_target(image: Image.Image, target_size_mb=1, min_quality=20, max_quality=95):
#     """
#     Compress PIL image to approximately target_size_mb
#     Returns a compressed PIL Image
#     """
#     target_bytes = target_size_mb * 1024 * 1024
#     quality = max_quality
#     img_bytes = io.BytesIO()

#     # Try iterative binary search approach to hit ~target size
#     while True:
#         img_bytes.seek(0)
#         image.save(img_bytes, format="JPEG", quality=quality, optimize=True)
#         size = img_bytes.tell()

#         if size <= target_bytes or quality <= min_quality:
#             break

#         # reduce quality
#         quality = int(quality * 0.85)
#         if quality < min_quality:
#             quality = min_quality

#     img_bytes.seek(0)
#     return Image.open(img_bytes)

# # ==========================================
# # 🚀 LOAD MODELS
# # ==========================================
# def load_models():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"🖥️ Device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

#     print("🔹 Loading SigLIP...")
#     siglip_proc = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
#     siglip_model = AutoModel.from_pretrained(
#         "google/siglip-so400m-patch14-384"
#     ).to(device)

#     print("🔹 Loading InstructBLIP...")
#     blip_proc = InstructBlipProcessor.from_pretrained(
#         "Salesforce/instructblip-flan-t5-xl"
#     )
#     blip_model = InstructBlipForConditionalGeneration.from_pretrained(
#         "Salesforce/instructblip-flan-t5-xl",
#         torch_dtype=torch.float16,
#         device_map="auto"
#     )

#     print("✅ Models loaded\n")
#     return device, siglip_proc, siglip_model, blip_proc, blip_model

# # ==========================================
# # 🧠 IMAGE PROCESSING
# # ==========================================
# def process_image(img_path, device, s_proc, s_model, b_proc, b_model):
#     try:
#         image = Image.open(img_path).convert("RGB")
#     except:
#         return None

#     # ---- Step 1: Compress image to ~1 MB ----
#     compressed_image = compress_image_to_target(image, target_size_mb=1)

#     # Save compressed copy (optional, for debugging)
#     compressed_path = os.path.join(COMPRESSED_DIR, os.path.basename(img_path))
#     compressed_image.save(compressed_path, format="JPEG", quality=85, optimize=True)

#     # ---- CATEGORY (SigLIP) ----
#     main_keys = list(TAXONOMY.keys())
#     inputs = s_proc(text=main_keys, images=compressed_image, return_tensors="pt", padding=True).to(device)

#     with torch.no_grad():
#         out = s_model(**inputs)

#     main_cat = main_keys[torch.sigmoid(out.logits_per_image).argmax().item()]

#     sub_keys = TAXONOMY[main_cat]
#     inputs_sub = s_proc(text=sub_keys, images=compressed_image, return_tensors="pt", padding=True).to(device)

#     with torch.no_grad():
#         out_sub = s_model(**inputs_sub)

#     sub_cat = sub_keys[torch.sigmoid(out_sub.logits_per_image).argmax().item()]

#     # ---- DESCRIPTION (InstructBLIP) ----
#     prompt = "Describe the image in detail including objects, style, lighting and mood."
#     inputs_blip = b_proc(images=compressed_image, text=prompt, return_tensors="pt").to(device, torch.float16)

#     with torch.no_grad():
#         gen = b_model.generate(**inputs_blip, max_new_tokens=90)

#     caption = b_proc.batch_decode(gen, skip_special_tokens=True)[0]

#     keywords = ", ".join(sorted(set([
#         w.strip(".,").lower()
#         for w in caption.split()
#         if len(w) > 4
#     ])))

#     return {
#         "filename": os.path.basename(img_path),
#         "image_path": f"compressed/{os.path.basename(img_path)}",
#         "main_cat": main_cat,
#         "sub_cat": sub_cat,
#         "caption": caption,
#         "keywords": keywords
#     }

# # ==========================================
# # 🧾 HTML GENERATOR
# # ==========================================
# def generate_html(results):
#     rows = ""
#     for r in results:
#         rows += f"""
#         <tr>
#             <td><img src="{r['image_path']}" width="120"></td>
#             <td>{r['main_cat']}</td>
#             <td>{r['sub_cat']}</td>
#             <td>{r['caption']}</td>
#             <td>{r['keywords']}</td>
#         </tr>
#         """

#     html = f"""
#     <!DOCTYPE html>
#     <html>
#     <head>
#         <meta charset="UTF-8">
#         <title>Image Metadata Results</title>
#         <style>
#             body {{ font-family: Arial, sans-serif; background:#f5f5f5; }}
#             table {{ border-collapse: collapse; width: 100%; background:white; }}
#             th, td {{ border: 1px solid #ddd; padding: 10px; vertical-align: top; }}
#             th {{ background: #222; color: white; }}
#             img {{ border-radius: 6px; }}
#         </style>
#     </head>
#     <body>
#         <h2>📸 Image Metadata Results</h2>
#         <table>
#             <tr>
#                 <th>Image</th>
#                 <th>Main Category</th>
#                 <th>Sub Category</th>
#                 <th>Description</th>
#                 <th>Keywords</th>
#             </tr>
#             {rows}
#         </table>
#     </body>
#     </html>
#     """

#     with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
#         f.write(html)

# # ==========================================
# # ▶️ MAIN
# # ==========================================
# if __name__ == "__main__":
#     device, s_proc, s_model, b_proc, b_model = load_models()

#     images = [
#         os.path.join(IMAGE_DIR, f)
#         for f in os.listdir(IMAGE_DIR)
#         if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
#     ]

#     print(f"📸 Found {len(images)} images\n")

#     results = []
#     for img in tqdm(images, desc="Processing Images"):
#         data = process_image(img, device, s_proc, s_model, b_proc, b_model)
#         if data:
#             results.append(data)

#     if results:
#         generate_html(results)
#         print(f"\n✅ DONE! HTML file created:\n{OUTPUT_HTML}")
#     else:
#         print("⚠️ No results generated.")



























































































import os
import io
import torch
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import (
    AutoModel,
    AutoProcessor,
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration
)

# ==========================================
# 📂 PATH SETUP
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")
OUTPUT_HTML = os.path.join(BASE_DIR, "metadata_results2.html")
COMPRESSED_DIR = os.path.join(BASE_DIR, "compressed")

os.makedirs(COMPRESSED_DIR, exist_ok=True)

if not os.path.exists(IMAGE_DIR):
    print("❌ 'images' folder not found next to main.py")
    exit()

# ==========================================
# 🧠 TAXONOMY
# ==========================================
TAXONOMY = {
    "Backgrounds": ["Abstract Pattern", "Texture", "Gradient Colors", "Bokeh Blur", "Geometric Shapes"],
    "Business": ["Office Meeting", "Financial Charts", "Handshake", "Working on Laptop", "Corporate Team"],
    "People": ["Portrait", "Crowd", "Family", "Fitness & Sports", "Happy Emotion"],
    "Technology": ["Artificial Intelligence", "Coding & Programming", "Virtual Reality", "Circuit Board", "Smart Devices"],
    "Travel": ["Mountains & Hiking", "Beach & Ocean", "Cityscape & Architecture", "Airport & Luggage", "Historical Monuments"],
    "Nature": ["Forest & Trees", "Sky & Clouds", "Flowers & Garden", "Desert & Sand", "Waterfalls & Rivers"],
    "Medical": ["Doctor & Nurse", "Hospital Bed", "Pills & Medicine", "Microscope & Lab", "Surgery"],
    "Food": ["Fresh Fruits", "Fast Food", "Desserts & Cakes", "Coffee & Drinks", "Healthy Salad"],
    "Sci-Fi & Fantasy": ["Space & Galaxy", "Cyberpunk & Futuristic", "Surrealism", "3D Render", "Aliens & Robots"],
    "Animals & Wildlife": ["Pets", "Wild Animals", "Birds", "Marine Life", "Insects", "Farm Animals"]
}

# ==========================================
# 🔧 IMAGE COMPRESSION FUNCTION (~1MB)
# ==========================================
def compress_image_to_target(image: Image.Image, target_size_mb=1, min_quality=20, max_quality=95):
    target_bytes = target_size_mb * 1024 * 1024
    quality = max_quality
    img_bytes = io.BytesIO()

    while True:
        img_bytes.seek(0)
        image.save(img_bytes, format="JPEG", quality=quality, optimize=True)
        size = img_bytes.tell()

        if size <= target_bytes or quality <= min_quality:
            break

        quality = int(quality * 0.85)
        if quality < min_quality:
            quality = min_quality

    img_bytes.seek(0)
    return Image.open(img_bytes)

# ==========================================
# 🚀 LOAD MODELS
# ==========================================
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

    print("🔹 Loading SigLIP...")
    siglip_proc = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    siglip_model = AutoModel.from_pretrained(
        "google/siglip-so400m-patch14-384"
    ).to(device)

    print("🔹 Loading InstructBLIP...")
    blip_proc = InstructBlipProcessor.from_pretrained(
        "Salesforce/instructblip-flan-t5-xl"
    )
    blip_model = InstructBlipForConditionalGeneration.from_pretrained(
        "Salesforce/instructblip-flan-t5-xl",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("✅ Models loaded\n")
    return device, siglip_proc, siglip_model, blip_proc, blip_model

# ==========================================
# 🧠 PROCESS SINGLE IMAGE
# ==========================================
def process_single_image(img_path, device, s_proc, s_model, b_proc, b_model):
    try:
        image = Image.open(img_path).convert("RGB")
    except:
        return None

    # ---- Step 1: Compress image to ~1 MB ----
    compressed_image = compress_image_to_target(image, target_size_mb=1)

    # Save compressed copy
    compressed_path = os.path.join(COMPRESSED_DIR, os.path.basename(img_path))
    compressed_image.save(compressed_path, format="JPEG", quality=85, optimize=True)

    # ---- CATEGORY (SigLIP) ----
    main_keys = list(TAXONOMY.keys())
    inputs = s_proc(text=main_keys, images=compressed_image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        out = s_model(**inputs)

    main_cat = main_keys[torch.sigmoid(out.logits_per_image).argmax().item()]

    sub_keys = TAXONOMY[main_cat]
    inputs_sub = s_proc(text=sub_keys, images=compressed_image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        out_sub = s_model(**inputs_sub)

    sub_cat = sub_keys[torch.sigmoid(out_sub.logits_per_image).argmax().item()]

    # ---- DESCRIPTION (InstructBLIP) ----
    prompt = "Describe the image in detail including objects, style, lighting and mood."
    inputs_blip = b_proc(images=compressed_image, text=prompt, return_tensors="pt").to(device, torch.float16)

    with torch.no_grad():
        gen = b_model.generate(**inputs_blip, max_new_tokens=90)

    caption = b_proc.batch_decode(gen, skip_special_tokens=True)[0]

    keywords = ", ".join(sorted(set([
        w.strip(".,").lower()
        for w in caption.split()
        if len(w) > 4
    ])))

    return {
        "filename": os.path.basename(img_path),
        "image_path": f"compressed/{os.path.basename(img_path)}",
        "main_cat": main_cat,
        "sub_cat": sub_cat,
        "caption": caption,
        "keywords": keywords
    }

# ==========================================
# 🧾 HTML GENERATOR
# ==========================================
def generate_html(results):
    rows = ""
    for r in results:
        rows += f"""
        <tr>
            <td><img src="{r['image_path']}" width="120"></td>
            <td>{r['main_cat']}</td>
            <td>{r['sub_cat']}</td>
            <td>{r['caption']}</td>
            <td>{r['keywords']}</td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Image Metadata Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; background:#f5f5f5; }}
            table {{ border-collapse: collapse; width: 100%; background:white; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; vertical-align: top; }}
            th {{ background: #222; color: white; }}
            img {{ border-radius: 6px; }}
        </style>
    </head>
    <body>
        <h2>📸 Image Metadata Results</h2>
        <table>
            <tr>
                <th>Image</th>
                <th>Main Category</th>
                <th>Sub Category</th>
                <th>Description</th>
                <th>Keywords</th>
            </tr>
            {rows}
        </table>
    </body>
    </html>
    """

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

# ==========================================
# ▶️ MAIN (MULTI-THREADING)
# ==========================================
if __name__ == "__main__":
    device, s_proc, s_model, b_proc, b_model = load_models()

    images = [
        os.path.join(IMAGE_DIR, f)
        for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    print(f"📸 Found {len(images)} images\n")

    results = []

    # ---- Use ThreadPoolExecutor for parallel image processing ----
    max_workers = min(8, os.cpu_count())  # adjust threads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_img = {
            executor.submit(process_single_image, img, device, s_proc, s_model, b_proc, b_model): img
            for img in images
        }

        for future in tqdm(as_completed(future_to_img), total=len(images), desc="Processing Images"):
            data = future.result()
            if data:
                results.append(data)

    if results:
        generate_html(results)
        print(f"\n✅ DONE! HTML file created:\n{OUTPUT_HTML}")
    else:
        print("⚠️ No results generated.")
