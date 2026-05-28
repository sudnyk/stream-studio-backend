from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from supabase import create_client
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from typing import Optional
import base64
import os
import uuid

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL is missing in .env")

if not SUPABASE_SERVICE_KEY:
    raise RuntimeError("SUPABASE_SERVICE_KEY is missing in .env")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

app = FastAPI(title="Stream Studio Backend")

PLANS = {
    "trial": {"max_channels": 1, "max_streams": 3, "ai_credits": 30, "days": 1},
    "starter": {"max_channels": 2, "max_streams": 10, "ai_credits": 500, "days": 30},
    "pro": {"max_channels": 3, "max_streams": 15, "ai_credits": 3000, "days": 30},
    "studio": {"max_channels": 4, "max_streams": 20, "ai_credits": 10000, "days": 30},
}

GUMROAD_PRODUCT_URLS = {
    "starter": "https://sudnyk.gumroad.com/l/zdmyrh",
    "pro": "",
    "studio": "",
}

AI_COSTS = {
    "generate_title": 1,
    "generate_seo_pack": 2,
    "generate_thumbnail_prompt": 1,
    "generate_thumbnail_image": 5,
}

PLAN_PRIORITY = {"studio": 4, "pro": 3, "starter": 2, "trial": 1, None: 0, "": 0}


class TrialRequest(BaseModel):
    hardware_id: str


class CheckoutRequest(BaseModel):
    hardware_id: str
    plan: str


class AIRequest(BaseModel):
    prompt: str = ""
    hardware_id: Optional[str] = None
    license_key: Optional[str] = None


def now_utc():
    return datetime.now(timezone.utc)


def iso(dt):
    return dt.isoformat()


def parse_datetime(value):
    if not value:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def is_not_expired(record):
    expires_at = record.get("expires_at")
    if not expires_at:
        return bool(record.get("is_active"))
    try:
        return parse_datetime(expires_at).timestamp() >= now_utc().timestamp()
    except Exception:
        return False


def choose_best_record(records):
    if not records:
        return None
    def score(r):
        active = 1 if bool(r.get("is_active")) and is_not_expired(r) else 0
        plan_score = PLAN_PRIORITY.get(str(r.get("plan", "")).lower(), 0)
        credits = int(r.get("ai_credits") or 0)
        return (active, plan_score, credits)
    return sorted(records, key=score, reverse=True)[0]


def get_license_by_hardware(hardware_id: str):
    result = supabase.table("licenses").select("*").eq("hardware_id", hardware_id).execute()
    return choose_best_record(result.data or [])


def get_license_by_email(email: str):
    result = supabase.table("licenses").select("*").eq("email", email).execute()
    return choose_best_record(result.data or [])


def get_license_by_key(license_key: str):
    result = supabase.table("licenses").select("*").eq("license_key", license_key).execute()
    return choose_best_record(result.data or [])


def build_status(record):
    expires_at = record.get("expires_at")
    expired = False
    if expires_at:
        exp = parse_datetime(expires_at)
        expired = exp.timestamp() < now_utc().timestamp()
    active = bool(record.get("is_active")) and not expired
    return {
        "active": active,
        "blocked": not active,
        "reason": None if active else "trial_or_plan_expired",
        "plan": record.get("plan"),
        "expires_at": expires_at,
        "max_channels": record.get("max_channels", 0),
        "max_streams": record.get("max_streams", 0),
        "ai_credits": record.get("ai_credits", 0),
        "credits_left": record.get("ai_credits", 0),
        "email": record.get("email"),
        "hardware_id": record.get("hardware_id"),
    }



def calculate_new_expiration(existing_record, plan_days):
    """
    Extends subscription correctly:
    - if current expires_at is still in the future, add days to that date;
    - if expired or missing, start from now.
    """
    base = now_utc()

    if existing_record and existing_record.get("expires_at"):
        try:
            current_exp = parse_datetime(existing_record.get("expires_at"))
            if current_exp.timestamp() > base.timestamp():
                base = current_exp
        except Exception:
            pass

    return base + timedelta(days=plan_days)

def update_license_credits(record, new_credits):
    # Prefer id when available; otherwise update by hardware_id.
    if record.get("id"):
        supabase.table("licenses").update({"ai_credits": new_credits}).eq("id", record["id"]).execute()
    else:
        supabase.table("licenses").update({"ai_credits": new_credits}).eq("hardware_id", record["hardware_id"]).execute()


def get_ai_license(req: AIRequest):
    if req.hardware_id:
        record = get_license_by_hardware(req.hardware_id)
        if record:
            return record
    if req.license_key:
        record = get_license_by_key(req.license_key)
        if record:
            return record
    return None


def charge_ai_credits(req: AIRequest, cost: int):
    record = get_ai_license(req)
    if not record and req.hardware_id:
        # If AI is called before /license/status, create trial automatically.
        status = start_trial(TrialRequest(hardware_id=req.hardware_id))
        record = get_license_by_hardware(req.hardware_id)
    if not record:
        return None, {"error": "License not found. Press Check Credits or restart the app.", "credits_left": 0}
    status = build_status(record)
    if status.get("blocked"):
        return None, {"error": "Subscription expired or blocked. Please upgrade your plan.", "credits_left": record.get("ai_credits", 0), "blocked": True}
    current_credits = int(record.get("ai_credits") or 0)
    if current_credits < cost:
        return None, {"error": f"Not enough AI credits. Required: {cost}, available: {current_credits}.", "credits_left": current_credits}
    new_credits = current_credits - cost
    update_license_credits(record, new_credits)
    record["ai_credits"] = new_credits
    return record, {"credits_left": new_credits, "ai_credits": new_credits, "charged": cost}


def make_demo_thumbnail_base64(title_text: str = "AI LIVESTREAM"):
    width, height = 1280, 720
    image = Image.new("RGB", (width, height), (10, 14, 24))
    draw = ImageDraw.Draw(image)
    for y in range(height):
        r = int(10 + y * 0.04)
        g = int(14 + y * 0.02)
        b = int(35 + y * 0.08)
        draw.line((0, y, width, y), fill=(r, g, min(b, 95)))
    draw.ellipse((760, 80, 1240, 560), fill=(35, 75, 135))
    draw.ellipse((820, 140, 1180, 500), fill=(70, 110, 180))
    draw.rectangle((0, 560, width, height), fill=(5, 8, 14))
    title = title_text.strip() or "AI LIVESTREAM"
    if len(title) > 42:
        title = title[:42] + "..."
    try:
        font_big = ImageFont.truetype("arial.ttf", 72)
        font_mid = ImageFont.truetype("arial.ttf", 42)
    except Exception:
        font_big = ImageFont.load_default()
        font_mid = ImageFont.load_default()
    draw.text((70, 130), "STREAM STUDIO AI", fill=(255, 255, 255), font=font_big)
    draw.text((72, 240), title.upper(), fill=(255, 220, 80), font=font_mid)
    draw.text((72, 610), "24/7 LIVE • SEO • THUMBNAILS • AUTOMATION", fill=(240, 240, 240), font=font_mid)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.get("/")
def home():
    return {"status": "Stream Studio Backend is running"}


@app.post("/trial/start")
def start_trial(req: TrialRequest):
    existing = get_license_by_hardware(req.hardware_id)
    if existing:
        return build_status(existing)
    plan = PLANS["trial"]
    started = now_utc()
    expires = started + timedelta(days=plan["days"])
    data = {
        "hardware_id": req.hardware_id,
        "license_key": "TRIAL-" + str(uuid.uuid4()).split("-")[0].upper(),
        "plan": "trial",
        "trial_started_at": iso(started),
        "expires_at": iso(expires),
        "is_active": True,
        "is_trial_used": True,
        "max_channels": plan["max_channels"],
        "max_streams": plan["max_streams"],
        "ai_credits": plan["ai_credits"],
    }
    result = supabase.table("licenses").insert(data).execute()
    return build_status(result.data[0])


@app.post("/license/status")
def license_status(req: TrialRequest):
    record = get_license_by_hardware(req.hardware_id)
    if not record:
        return start_trial(req)
    return build_status(record)


@app.post("/billing/create-checkout")
def create_checkout(req: CheckoutRequest):
    plan = req.plan.lower().strip()
    if plan not in ["starter", "pro", "studio"]:
        raise HTTPException(status_code=400, detail="Invalid plan")
    url = GUMROAD_PRODUCT_URLS.get(plan)
    if not url:
        raise HTTPException(status_code=400, detail=f"Gumroad URL for {plan} is not configured in backend.")
    return {"checkout_url": url}


@app.post("/webhooks/gumroad")
async def gumroad_webhook(request: Request):
    form = await request.form()
    data = dict(form)
    print("GUMROAD WEBHOOK:", data)
    email = str(data.get("email", "")).lower().strip()
    product_name = str(data.get("product_name", "")).lower().strip()
    subscription_id = str(data.get("subscription_id", "")).strip()
    refunded = str(data.get("refunded", "false")).lower() == "true"
    disputed = str(data.get("disputed", "false")).lower() == "true"
    if "starter" in product_name:
        plan_name = "starter"
    elif "pro" in product_name:
        plan_name = "pro"
    elif "studio" in product_name:
        plan_name = "studio"
    else:
        return {"status": "unknown_product", "product_name": product_name}
    if not email:
        return {"status": "missing_email"}
    existing = get_license_by_email(email)
    if refunded or disputed:
        if existing:
            supabase.table("licenses").update({"is_active": False, "plan": "expired"}).eq("email", email).execute()
        return {"status": "blocked_refund_or_dispute", "email": email}
    plan = PLANS[plan_name]
    expires = calculate_new_expiration(existing, plan["days"])
    update_data = {
        "email": email,
        "plan": plan_name,
        "expires_at": iso(expires),
        "is_active": True,
        "max_channels": plan["max_channels"],
        "max_streams": plan["max_streams"],
        "ai_credits": plan["ai_credits"],
        "gumroad_subscription_id": subscription_id,
        "gumroad_product_name": data.get("product_name"),
        "gumroad_sale_id": data.get("sale_id"),
        "last_payment_at": iso(now_utc()),
    }
    if existing:
        supabase.table("licenses").update(update_data).eq("email", email).execute()
    else:
        update_data["hardware_id"] = "EMAIL-" + email
        update_data["license_key"] = "GUM-" + str(uuid.uuid4()).split("-")[0].upper()
        update_data["is_trial_used"] = True
        supabase.table("licenses").insert(update_data).execute()
    return {"status": "ok", "email": email, "plan": plan_name, "expires_at": iso(expires)}


@app.get("/payment-success")
def payment_success():
    return {"message": "Payment successful. You can return to the app."}


@app.get("/payment-cancel")
def payment_cancel():
    return {"message": "Payment canceled."}


@app.post("/generate-title")
def generate_title(req: AIRequest):
    print("AI /generate-title", req.dict())
    record, meta = charge_ai_credits(req, AI_COSTS["generate_title"])
    if not record:
        return meta
    title = "🔥 Relaxing 24/7 AI Livestream | Calm Ambience, Sleep, Study & Focus"
    return {**meta, "result": title, "title": title}


@app.post("/generate-seo-pack")
def generate_seo_pack(req: AIRequest):
    print("AI /generate-seo-pack", req.dict())
    record, meta = charge_ai_credits(req, AI_COSTS["generate_seo_pack"])
    if not record:
        return meta
    title = "Relaxing 24/7 AI Livestream for Sleep, Study and Focus"
    description = "Enjoy a relaxing 24/7 livestream with calming ambience, peaceful visuals, and soothing atmosphere for sleep, study, relaxation and focus."
    tags = ["relaxing livestream", "sleep music", "study ambience", "focus music", "24/7 live", "calm ambience", "youtube live"]
    hashtags = ["#livestream", "#sleep", "#study", "#relaxing", "#focus"]
    return {**meta, "result": f"TITLE:\n{title}\n\nDESCRIPTION:\n{description}\n\nTAGS:\n{', '.join(tags)}\n\nHASHTAGS:\n{' '.join(hashtags)}", "title": title, "description": description, "tags": tags, "hashtags": hashtags}


@app.post("/generate-thumbnail-prompt")
def generate_thumbnail_prompt(req: AIRequest):
    print("AI /generate-thumbnail-prompt", req.dict())
    record, meta = charge_ai_credits(req, AI_COSTS["generate_thumbnail_prompt"])
    if not record:
        return meta
    prompt = "Ultra-realistic cinematic YouTube thumbnail, cozy livestream atmosphere, dramatic lighting, high contrast, eye-catching composition, no text, 16:9"
    return {**meta, "result": prompt, "prompt": prompt, "thumbnail_prompt": prompt}


@app.post("/generate-thumbnail-image")
def generate_thumbnail_image(req: AIRequest):
    print("AI /generate-thumbnail-image", req.dict())
    record, meta = charge_ai_credits(req, AI_COSTS["generate_thumbnail_image"])
    if not record:
        return meta
    image_base64 = make_demo_thumbnail_base64(req.prompt)
    return {**meta, "result": "Thumbnail image generated successfully.", "image_base64": image_base64, "filename": "thumbnail.png", "message": "Thumbnail image generated successfully."}
