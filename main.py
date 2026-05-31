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
ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY", "").strip()

if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL is missing in .env")

if not SUPABASE_SERVICE_KEY:
    raise RuntimeError("SUPABASE_SERVICE_KEY is missing in .env")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# DEV fallback is only for local development/testing.
# When Supabase DNS/internet is unavailable, the backend will not crash.
# Clean local fallback is Trial only: 1 channel, 3 streams, 30 AI credits.
# For production on Render, set DEV_OFFLINE_LICENSE=false in environment variables.
# Production-safe default: offline fallback is OFF unless you explicitly enable it in local .env.
DEV_OFFLINE_LICENSE = os.getenv("DEV_OFFLINE_LICENSE", "false").strip().lower() in ("1", "true", "yes", "on")


def make_dev_offline_license(hardware_id: str = "", email: str = "", license_key: str = ""):
    """
    Clean-client safe local fallback.

    Important:
    This fallback must NEVER unlock Studio/Pro features.
    It only lets the desktop app open during local development if Supabase is offline.
    Real paid plans must come only from Supabase / Gumroad / admin endpoints.
    """
    plan = PLANS["trial"]
    expires = now_utc() + timedelta(days=plan["days"])
    return {
        "id": "DEV-OFFLINE-TRIAL",
        "hardware_id": hardware_id or "DEV-HARDWARE",
        "email": email or "",
        "license_key": license_key or "DEV-OFFLINE-TRIAL",
        "plan": "trial",
        "expires_at": iso(expires),
        "is_active": True,
        "max_channels": plan["max_channels"],
        "max_streams": plan["max_streams"],
        "ai_credits": plan["ai_credits"],
        "credits_left": plan["ai_credits"],
        "dev_offline": True,
        "clean_client_safe": True,
    }


def should_use_dev_offline_fallback():
    return bool(DEV_OFFLINE_LICENSE)


def log_supabase_error(place: str, error: Exception):
    print(f"[SUPABASE OFFLINE] {place}: {error}")


app = FastAPI(title="Stream Studio Backend")

PLANS = {
    "trial": {"max_channels": 1, "max_streams": 3, "ai_credits": 30, "days": 1},
    "starter": {"max_channels": 2, "max_streams": 10, "ai_credits": 500, "days": 30},
    "pro": {"max_channels": 3, "max_streams": 15, "ai_credits": 3000, "days": 30},
    "studio": {"max_channels": 4, "max_streams": 20, "ai_credits": 10000, "days": 30},
}

GUMROAD_PRODUCT_URLS = {
    "starter": "https://sudnyk.gumroad.com/l/zdmyrh",
    "pro": os.getenv("GUMROAD_PRO_URL", ""),
    "studio": os.getenv("GUMROAD_STUDIO_URL", ""),

    # One-time AI credit top-up products.
    # Add these URLs in Render Environment Variables after creating Gumroad products:
    # CREDITS_1000_URL, CREDITS_5000_URL, CREDITS_15000_URL
    "credits_1000": os.getenv("CREDITS_1000_URL", ""),
    "credits_5000": os.getenv("CREDITS_5000_URL", ""),
    "credits_15000": os.getenv("CREDITS_15000_URL", ""),
}

CREDIT_PACKS = {
    "credits_1000": 1000,
    "credits_5000": 5000,
    "credits_15000": 15000,
}


AI_COSTS = {
    "generate_title": 1,
    "generate_seo_pack": 2,
    "generate_thumbnail_prompt": 1,
    "generate_thumbnail_image": 5,
}

PLAN_PRIORITY = {"studio": 4, "pro": 3, "starter": 2, "trial": 1, None: 0, "": 0}


class TrialRequest(BaseModel):
    hardware_id: str = ""
    email: Optional[str] = ""
    license_key: Optional[str] = ""


class CheckoutRequest(BaseModel):
    hardware_id: str
    plan: str


class AdminGrantPlanRequest(BaseModel):
    admin_secret: str
    email: str
    plan: str
    hardware_id: str = ""
    days: int = 30
    ai_credits: int | None = None


class AdminAddCreditsRequest(BaseModel):
    admin_secret: str
    email: str
    credits: int


class AdminExtendLicenseRequest(BaseModel):
    admin_secret: str
    email: str
    days: int


class AdminDisableLicenseRequest(BaseModel):
    admin_secret: str
    email: str
    reason: str = "manual_admin_disable"


class AdminResetHardwareRequest(BaseModel):
    admin_secret: str
    hardware_id: str
    delete_records: bool = True
    reason: str = "manual_trial_reset"






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
    try:
        result = supabase.table("licenses").select("*").eq("hardware_id", hardware_id).execute()
        return choose_best_record(result.data or [])
    except Exception as e:
        log_supabase_error("get_license_by_hardware", e)
        if should_use_dev_offline_fallback():
            return make_dev_offline_license(hardware_id=hardware_id)
        raise


def get_license_by_email(email: str):
    try:
        result = supabase.table("licenses").select("*").eq("email", email).execute()
        return choose_best_record(result.data or [])
    except Exception as e:
        log_supabase_error("get_license_by_email", e)
        if should_use_dev_offline_fallback():
            return make_dev_offline_license(email=email)
        raise


def get_license_by_key(license_key: str):
    try:
        result = supabase.table("licenses").select("*").eq("license_key", license_key).execute()
        return choose_best_record(result.data or [])
    except Exception as e:
        log_supabase_error("get_license_by_key", e)
        if should_use_dev_offline_fallback():
            return make_dev_offline_license(license_key=license_key)
        raise



def get_best_license_for_status(email: str = "", hardware_id: str = "", license_key: str = ""):
    """
    Status lookup must not let a trial hardware_id override a paid email license.
    It collects all possible matching records and chooses the best by:
    active/not expired, plan priority, credits.
    Priority: studio > pro > starter > trial.

    Local DEV fallback:
    If Supabase is temporarily unavailable, return a local offline Studio license
    instead of crashing the backend with 500.
    """
    records = []

    email = (email or "").lower().strip()
    hardware_id = (hardware_id or "").strip()
    license_key = (license_key or "").strip()
    supabase_failed = False

    try:
        if email:
            res = supabase.table("licenses").select("*").eq("email", email).execute()
            records.extend(res.data or [])
    except Exception as e:
        supabase_failed = True
        log_supabase_error("get_best_license_for_status/email", e)

    try:
        if hardware_id:
            res = supabase.table("licenses").select("*").eq("hardware_id", hardware_id).execute()
            records.extend(res.data or [])
    except Exception as e:
        supabase_failed = True
        log_supabase_error("get_best_license_for_status/hardware", e)

    try:
        if license_key:
            res = supabase.table("licenses").select("*").eq("license_key", license_key).execute()
            records.extend(res.data or [])
    except Exception as e:
        supabase_failed = True
        log_supabase_error("get_best_license_for_status/key", e)

    if supabase_failed and not records and should_use_dev_offline_fallback():
        return make_dev_offline_license(
            hardware_id=hardware_id,
            email=email,
            license_key=license_key,
        )

    # Deduplicate by id when possible, otherwise by email/hardware/plan/expires.
    deduped = []
    seen = set()

    for record in records:
        key = record.get("id") or (
            record.get("email"),
            record.get("hardware_id"),
            record.get("plan"),
            record.get("expires_at"),
        )

        key = str(key)

        if key in seen:
            continue

        seen.add(key)
        deduped.append(record)

    return choose_best_record(deduped)


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
        "license_key": record.get("license_key"),
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
    """
    Store new AI credits in Supabase and return the updated license row when possible.

    Important for V15.2:
    The desktop app trusts credits_left returned by the backend.
    So after charging we must update Supabase and then verify/read the new value back.
    """
    new_credits = int(new_credits)

    # Offline DEV records are not stored in Supabase.
    if record.get("dev_offline"):
        record["ai_credits"] = new_credits
        return record

    try:
        update_data = {
            "ai_credits": new_credits,
        }

        # Prefer id when available. This is the safest because hardware_id/email can be duplicated.
        if record.get("id"):
            result = (
                supabase.table("licenses")
                .update(update_data)
                .eq("id", record["id"])
                .execute()
            )
        elif record.get("license_key"):
            result = (
                supabase.table("licenses")
                .update(update_data)
                .eq("license_key", record["license_key"])
                .execute()
            )
        elif record.get("hardware_id"):
            result = (
                supabase.table("licenses")
                .update(update_data)
                .eq("hardware_id", record["hardware_id"])
                .execute()
            )
        elif record.get("email"):
            result = (
                supabase.table("licenses")
                .update(update_data)
                .eq("email", record["email"])
                .execute()
            )
        else:
            raise RuntimeError("Cannot update credits: license row has no id/license_key/hardware_id/email")

        updated_rows = getattr(result, "data", None) or []
        if updated_rows:
            return updated_rows[0]

        # Some Supabase/PostgREST settings may return no rows after update.
        # In that case, return local record with the new credits so UI still updates.
        record["ai_credits"] = new_credits
        return record

    except Exception as e:
        log_supabase_error("update_license_credits", e)
        if not should_use_dev_offline_fallback():
            raise
        record["ai_credits"] = new_credits
        record["dev_offline"] = True
        return record


def refresh_ai_license_record(record):
    """Read the charged license again from Supabase, when possible."""
    if not record or record.get("dev_offline"):
        return record

    try:
        if record.get("id"):
            res = supabase.table("licenses").select("*").eq("id", record["id"]).execute()
            rows = res.data or []
            return rows[0] if rows else record
        if record.get("license_key"):
            refreshed = get_license_by_key(record["license_key"])
            return refreshed or record
        if record.get("hardware_id"):
            refreshed = get_license_by_hardware(record["hardware_id"])
            return refreshed or record
        if record.get("email"):
            refreshed = get_license_by_email(record["email"])
            return refreshed or record
    except Exception as e:
        log_supabase_error("refresh_ai_license_record", e)

    return record


def get_ai_license(req: AIRequest):
    # IMPORTANT: paid license key must have priority over hardware_id.
    # Otherwise AI generation can charge an old local trial record instead of the paid Starter/Pro/Studio license.
    if req.license_key:
        record = get_license_by_key(req.license_key)
        if record:
            return record
    if req.hardware_id:
        record = get_license_by_hardware(req.hardware_id)
        if record:
            return record
    return None


def charge_ai_credits(req: AIRequest, cost: int):
    """
    Deduct AI credits for Stream Manager AI mode.

    V15.2 behavior:
    - license_key has priority over hardware_id;
    - if no license exists, trial is created automatically;
    - credits are updated in Supabase;
    - backend returns previous_credits, credits_left and charged so the desktop UI can update immediately.
    """
    cost = int(cost)
    record = get_ai_license(req)

    if not record and req.hardware_id:
        # If AI is called before /license/status, create trial automatically.
        trial_status = start_trial(TrialRequest(hardware_id=req.hardware_id, license_key=req.license_key or ""))
        if isinstance(trial_status, dict) and trial_status.get("license_key"):
            # Read the inserted trial again so we have id/license_key and update exactly that row.
            record = get_license_by_key(trial_status.get("license_key")) or get_license_by_hardware(req.hardware_id)
            if not record:
                record = {
                    "hardware_id": trial_status.get("hardware_id"),
                    "email": trial_status.get("email"),
                    "license_key": trial_status.get("license_key"),
                    "plan": trial_status.get("plan"),
                    "expires_at": trial_status.get("expires_at"),
                    "is_active": trial_status.get("active", True),
                    "max_channels": trial_status.get("max_channels", 0),
                    "max_streams": trial_status.get("max_streams", 0),
                    "ai_credits": trial_status.get("ai_credits", trial_status.get("credits_left", 0)),
                    "dev_offline": bool(trial_status.get("dev_offline")),
                }
        else:
            record = get_license_by_hardware(req.hardware_id)

    if not record:
        return None, {"error": "License not found. Press Check Credits or restart the app.", "credits_left": 0}

    status = build_status(record)

    if status.get("blocked"):
        return None, {
            "error": "Subscription expired or blocked. Please upgrade your plan.",
            "credits_left": int(record.get("ai_credits") or 0),
            "blocked": True,
        }

    current_credits = int(record.get("ai_credits") or 0)

    if current_credits < cost:
        return None, {
            "error": f"Not enough AI credits. Required: {cost}, available: {current_credits}.",
            "credits_left": current_credits,
            "ai_credits": current_credits,
            "charged": 0,
        }

    new_credits = current_credits - cost
    updated_record = update_license_credits(record, new_credits) or record
    updated_record = refresh_ai_license_record(updated_record) or updated_record

    verified_credits = int(updated_record.get("ai_credits") if updated_record.get("ai_credits") is not None else new_credits)

    meta = {
        "credits_left": verified_credits,
        "ai_credits": verified_credits,
        "previous_credits": current_credits,
        "charged": cost,
        "plan": updated_record.get("plan"),
        "license_key": updated_record.get("license_key"),
    }

    if updated_record.get("dev_offline"):
        meta["dev_offline"] = True
        meta["warning"] = "DEV OFFLINE MODE: Supabase is unavailable, local test license is being used."

    print(
        f"AI CREDIT CHARGE OK | plan={meta.get('plan')} "
        f"license={str(meta.get('license_key') or '')[:18]} "
        f"cost={cost} before={current_credits} after={verified_credits}"
    )

    return updated_record, meta


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



def require_admin_secret(admin_secret: str):
    if not ADMIN_SECRET_KEY:
        raise HTTPException(
            status_code=500,
            detail="ADMIN_SECRET_KEY is not configured on backend."
        )

    if not admin_secret or admin_secret != ADMIN_SECRET_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid admin secret."
        )


def normalize_email(email: str):
    email = (email or "").lower().strip()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Valid email is required.")
    return email


def normalize_plan(plan: str):
    plan = (plan or "").lower().strip()
    if plan not in PLANS:
        raise HTTPException(status_code=400, detail=f"Invalid plan: {plan}")
    return plan


def get_plan_default_credits(plan: str):
    plan_config = PLANS.get(plan, {})
    return int(plan_config.get("ai_credits", 0) or 0)


def admin_get_license_by_email_or_error(email: str):
    record = get_license_by_email(email)
    if not record:
        raise HTTPException(status_code=404, detail=f"No license found for {email}")
    return record


def update_license_by_existing_record(existing: dict, update_data: dict):
    if existing.get("id"):
        supabase.table("licenses").update(update_data).eq("id", existing["id"]).execute()
    else:
        supabase.table("licenses").update(update_data).eq("email", existing["email"]).execute()



@app.post("/trial/start")
def start_trial(req: TrialRequest):
    try:
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

    except Exception as e:
        log_supabase_error("start_trial", e)
        if should_use_dev_offline_fallback():
            return build_status(make_dev_offline_license(hardware_id=req.hardware_id, email=req.email or "", license_key=req.license_key or ""))
        raise


@app.post("/license/status")
def license_status(req: TrialRequest):
    # IMPORTANT:
    # If both email and hardware_id are present, choose the best license.
    # This prevents an old trial hardware record from overriding a paid email license.
    record = get_best_license_for_status(
        email=getattr(req, "email", ""),
        hardware_id=getattr(req, "hardware_id", ""),
        license_key=getattr(req, "license_key", "")
    )

    if not record:
        return start_trial(req)

    return build_status(record)


@app.post("/admin/debug-license")
def admin_debug_license(req: TrialRequest):
    """
    Quick admin/debug endpoint for checking the exact license row used by the app.
    It does not change anything and does not require admin secret.
    Use only for troubleshooting.
    """
    record = get_best_license_for_status(
        email=getattr(req, "email", ""),
        hardware_id=getattr(req, "hardware_id", ""),
        license_key=getattr(req, "license_key", "")
    )
    if not record:
        return {"found": False, "credits_left": 0}
    status = build_status(record)
    return {"found": True, **status}



@app.post("/admin/grant-plan")
def admin_grant_plan(req: AdminGrantPlanRequest):
    require_admin_secret(req.admin_secret)

    email = normalize_email(req.email)
    plan = normalize_plan(req.plan)

    days = int(req.days or 30)
    if days <= 0:
        raise HTTPException(status_code=400, detail="days must be greater than 0")

    plan_config = PLANS[plan]
    existing = get_license_by_email(email)

    now = now_utc()
    expires_at = now + timedelta(days=days)

    if req.ai_credits is None:
        credits = get_plan_default_credits(plan)
    else:
        credits = int(req.ai_credits)
        if credits < 0:
            raise HTTPException(status_code=400, detail="ai_credits cannot be negative")

    license_key = f"ADMIN-{plan.upper()}-{uuid.uuid4().hex[:12].upper()}"

    data = {
        "email": email,
        "license_key": license_key,
        "plan": plan,
        "hardware_id": req.hardware_id or (existing.get("hardware_id") if existing else ""),
        "expires_at": iso(expires_at),
        "is_active": True,
        "ai_credits": credits,
        "max_channels": int(plan_config.get("max_channels", 1)),
        "max_streams": int(plan_config.get("max_streams", 1)),
        "gumroad_subscription_id": f"MANUAL-{uuid.uuid4().hex[:10].upper()}",
        "gumroad_sale_id": f"ADMIN-GRANT-{uuid.uuid4().hex[:10].upper()}",
        "last_payment_at": iso(now),
    }

    if existing:
        update_license_by_existing_record(existing, data)
        action = "updated_existing_license"
    else:
        supabase.table("licenses").insert(data).execute()
        action = "created_new_license"

    return {
        "status": "ok",
        "action": action,
        "email": email,
        "license_key": data["license_key"],
        "plan": plan,
        "expires_at": data["expires_at"],
        "ai_credits": credits,
        "max_channels": data["max_channels"],
        "max_streams": data["max_streams"],
    }


@app.post("/admin/add-credits")
def admin_add_credits(req: AdminAddCreditsRequest):
    require_admin_secret(req.admin_secret)

    email = normalize_email(req.email)
    credits_to_add = int(req.credits)

    if credits_to_add <= 0:
        raise HTTPException(status_code=400, detail="credits must be greater than 0")

    existing = admin_get_license_by_email_or_error(email)

    current_credits = int(existing.get("ai_credits") or 0)
    new_credits = current_credits + credits_to_add

    update_data = {
        "ai_credits": new_credits,
        "gumroad_sale_id": f"ADMIN-CREDITS-{uuid.uuid4().hex[:10].upper()}",
        "last_payment_at": iso(now_utc()),
    }

    update_license_by_existing_record(existing, update_data)

    return {
        "status": "ok",
        "email": email,
        "added_credits": credits_to_add,
        "previous_credits": current_credits,
        "credits_left": new_credits,
    }


@app.post("/admin/extend-license")
def admin_extend_license(req: AdminExtendLicenseRequest):
    require_admin_secret(req.admin_secret)

    email = normalize_email(req.email)
    days = int(req.days)

    if days <= 0:
        raise HTTPException(status_code=400, detail="days must be greater than 0")

    existing = admin_get_license_by_email_or_error(email)

    old_expires = existing.get("expires_at")
    base_time = now_utc()

    if old_expires:
        try:
            parsed = datetime.fromisoformat(str(old_expires).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            if parsed > base_time:
                base_time = parsed
        except Exception:
            pass

    new_expires = base_time + timedelta(days=days)

    update_data = {
        "expires_at": iso(new_expires),
        "is_active": True,
        "gumroad_sale_id": f"ADMIN-EXTEND-{uuid.uuid4().hex[:10].upper()}",
        "last_payment_at": iso(now_utc()),
    }

    update_license_by_existing_record(existing, update_data)

    return {
        "status": "ok",
        "email": email,
        "plan": existing.get("plan"),
        "old_expires_at": old_expires,
        "new_expires_at": update_data["expires_at"],
        "added_days": days,
    }


@app.post("/admin/disable-license")
def admin_disable_license(req: AdminDisableLicenseRequest):
    require_admin_secret(req.admin_secret)

    email = normalize_email(req.email)
    existing = admin_get_license_by_email_or_error(email)

    update_data = {
        "is_active": False,
        "gumroad_sale_id": f"ADMIN-DISABLE-{uuid.uuid4().hex[:10].upper()}",
        "last_payment_at": iso(now_utc()),
    }

    update_license_by_existing_record(existing, update_data)

    return {
        "status": "ok",
        "email": email,
        "plan": existing.get("plan"),
        "is_active": False,
        "reason": req.reason,
    }




@app.post("/admin/reset-license-by-hardware")
def admin_reset_license_by_hardware(req: AdminResetHardwareRequest):
    require_admin_secret(req.admin_secret)

    hardware_id = (req.hardware_id or "").strip()
    if not hardware_id:
        raise HTTPException(status_code=400, detail="hardware_id is required")

    existing = supabase.table("licenses").select("*").eq("hardware_id", hardware_id).execute()
    records = existing.data or []

    if not records:
        return {
            "status": "ok",
            "action": "nothing_found",
            "hardware_id": hardware_id,
            "deleted_or_disabled": 0,
            "reason": req.reason,
        }

    count = 0

    for record in records:
        if req.delete_records:
            if record.get("id"):
                supabase.table("licenses").delete().eq("id", record["id"]).execute()
            else:
                supabase.table("licenses").delete().eq("hardware_id", hardware_id).execute()
            count += 1
        else:
            update_data = {
                "is_active": False,
                "gumroad_sale_id": f"ADMIN-RESET-{uuid.uuid4().hex[:10].upper()}",
                "last_payment_at": iso(now_utc()),
            }
            if record.get("id"):
                supabase.table("licenses").update(update_data).eq("id", record["id"]).execute()
            else:
                supabase.table("licenses").update(update_data).eq("hardware_id", hardware_id).execute()
            count += 1

    return {
        "status": "ok",
        "action": "deleted_records" if req.delete_records else "disabled_records",
        "hardware_id": hardware_id,
        "deleted_or_disabled": count,
        "reason": req.reason,
    }



@app.post("/billing/create-checkout")
def create_checkout(req: CheckoutRequest):
    plan = req.plan.lower().strip()

    allowed_products = ["starter", "pro", "studio", "credits_1000", "credits_5000", "credits_15000"]

    if plan not in allowed_products:
        raise HTTPException(status_code=400, detail="Invalid product")

    url = GUMROAD_PRODUCT_URLS.get(plan)

    if not url:
        raise HTTPException(
            status_code=400,
            detail=f"Gumroad URL for {plan} is not configured in backend."
        )

    return {"checkout_url": url, "product": plan}


def detect_credit_pack(product_name: str):
    name = (product_name or "").lower()

    if "credit" not in name and "credits" not in name:
        return None, 0

    if "15000" in name or "15,000" in name:
        return "credits_15000", 15000

    if "5000" in name or "5,000" in name:
        return "credits_5000", 5000

    if "1000" in name or "1,000" in name:
        return "credits_1000", 1000

    return None, 0


def add_credit_pack_to_license(email: str, add_credits: int, sale_id: str):
    existing = get_license_by_email(email)

    if not existing:
        return {
            "status": "no_license_for_credit_pack",
            "email": email,
            "message": "Credit packs require an existing trial or subscription license."
        }

    # Basic duplicate protection: if Gumroad retries the same sale_id, do not add credits twice.
    if sale_id and existing.get("gumroad_sale_id") == sale_id:
        return {
            "status": "already_processed",
            "email": email,
            "credits_left": existing.get("ai_credits", 0),
            "sale_id": sale_id,
        }

    current_credits = int(existing.get("ai_credits") or 0)
    new_credits = current_credits + int(add_credits)

    update_data = {
        "ai_credits": new_credits,
        "gumroad_sale_id": sale_id,
        "last_payment_at": iso(now_utc()),
    }

    if existing.get("id"):
        supabase.table("licenses").update(update_data).eq("id", existing["id"]).execute()
    else:
        supabase.table("licenses").update(update_data).eq("email", email).execute()

    return {
        "status": "credits_added",
        "email": email,
        "added_credits": add_credits,
        "credits_left": new_credits,
        "sale_id": sale_id,
    }



@app.post("/webhooks/gumroad")
async def gumroad_webhook(request: Request):
    form = await request.form()
    data = dict(form)
    print("GUMROAD WEBHOOK:", data)
    email = str(data.get("email", "")).lower().strip()
    product_name = str(data.get("product_name", "")).lower().strip()
    subscription_id = str(data.get("subscription_id", "")).strip()
    sale_id = str(data.get("sale_id", "")).strip()
    refunded = str(data.get("refunded", "false")).lower() == "true"
    disputed = str(data.get("disputed", "false")).lower() == "true"

    if not email:
        return {"status": "missing_email"}

    credit_pack_key, add_credits = detect_credit_pack(product_name)

    if credit_pack_key:
        if refunded or disputed:
            return {
                "status": "credit_pack_refund_or_dispute_received",
                "email": email,
                "credit_pack": credit_pack_key,
            }

        return add_credit_pack_to_license(email, add_credits, sale_id)

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
        "gumroad_sale_id": sale_id,
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
