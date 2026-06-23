import sys
import os

# Auto-relaunch inside virtual environment if dependencies are missing globally
try:
    import numpy as np
    import librosa
    import joblib
    import tensorflow as tf
    import fastapi
    import uvicorn
except ImportError:
    project_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(project_dir, "venv", "bin", "python")
    if not os.path.exists(venv_python):
        venv_python = os.path.join(project_dir, "myenv", "bin", "python")
        if not os.path.exists(venv_python):
            # Check for Windows virtualenv path just in case
            venv_python = os.path.join(project_dir, "venv", "Scripts", "python.exe")
            if not os.path.exists(venv_python):
                venv_python = os.path.join(project_dir, "myenv", "Scripts", "python.exe")
                
    if os.path.exists(venv_python):
        print(f"⚠️ Missing global dependencies. Relaunching program using virtual environment: {venv_python}")
        import subprocess
        # Replace current process with venv python execution
        if hasattr(os, 'execv'):
            os.execv(venv_python, [venv_python] + sys.argv)
        else:
            sys.exit(subprocess.run([venv_python] + sys.argv).returncode)
    else:
        print("❌ Error: Core dependencies (numpy, librosa, tensorflow, joblib) not found, and no local virtualenv 'venv' or 'myenv' was detected.")
        sys.exit(1)

import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Normal global imports to satisfy IDE static checkers and prevent redlines
import numpy as np
import librosa
import joblib
import tensorflow as tf
import uvicorn

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.saving import register_keras_serializable
from pydantic import BaseModel

# Initialize FastAPI App
app = FastAPI(title="Heart Failure & Stress Assessment API")

# Enable CORS for local development/testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- FIX: Register the Custom Layer for Keras 3 ---
@register_keras_serializable(package='Custom', name='LSTM')
class CustomLSTM(LSTM):
    @classmethod
    def from_config(cls, config):
        if 'time_major' in config:
            del config['time_major']
        if 'implementation' in config and config['implementation'] == 0:
             config['implementation'] = 1
        return super().from_config(config)

# --- Model Loading ---
try:
    heart_model = load_model(
        "heart_sounds.h5", 
        custom_objects={
            'LSTM': CustomLSTM, 
            'Bidirectional': Bidirectional
        }
    )
    stress_model = joblib.load("stress_model.pkl")
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    raise RuntimeError(f"Could not load models: {e}")

stress_mapping = {0: "Low", 1: "Moderate", 2: "High"}

# --- Preprocessing Function ---
def preprocess_audio(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_file_path = tmp_file.name

    try:
        audio, sr = librosa.load(tmp_file_path, sr=22050)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=25)
        mfccs = np.mean(mfccs.T, axis=0)
        return mfccs.reshape(1, 25, 1)
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

def calculate_heart_failure_risk(murmur_prob, age):
    base_risk = murmur_prob * 0.7
    age_risk = max(0, (age - 30) * 0.1)
    total_risk = base_risk + age_risk
    total_risk = max(0, min(100, total_risk))
    return total_risk

# --- Pydantic Schemas ---
class StressInput(BaseModel):
    anxiety_level: float
    mental_health_history: int  # 0 or 1
    depression: float
    headache: float
    sleep_quality: float
    breathing_problem: float
    living_conditions: float

# --- Routes ---

@app.post("/api/predict/heart")
async def predict_heart(
    file: UploadFile = File(...),
    age: int = Form(...)
):
    if not file.filename.lower().endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only WAV files are supported.")
    
    try:
        file_bytes = await file.read()
        data = preprocess_audio(file_bytes)
        predictions = heart_model.predict(data)
        
        artifact_prob = float(predictions[0][0] * 100)
        murmur_prob = float(predictions[0][1] * 100)
        normal_prob = float(predictions[0][2] * 100)
        
        risk_percentage = calculate_heart_failure_risk(murmur_prob, age)
        
        if risk_percentage >= 70:
            status = "critical"
            message = "Very high risk of heart failure. Seek immediate medical attention."
        elif risk_percentage >= 40:
            status = "warning"
            message = "Moderate to high risk of heart failure. Consult a cardiologist."
        elif risk_percentage >= 20:
            status = "info"
            message = "Low to moderate risk of heart failure. Monitor your health."
        else:
            status = "success"
            message = "Low risk of heart failure. No immediate concern."
            
        return {
            "artifact_probability": artifact_prob,
            "murmur_probability": murmur_prob,
            "normal_probability": normal_prob,
            "heart_failure_risk": risk_percentage,
            "status": status,
            "message": message
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing heart sound: {str(e)}")

@app.post("/api/predict/stress")
async def predict_stress(data: StressInput):
    try:
        X_stress = np.array([[
            data.anxiety_level,
            data.mental_health_history,
            data.depression,
            data.headache,
            data.sleep_quality,
            data.breathing_problem,
            data.living_conditions
        ]], dtype=float)
        
        predicted_class = int(stress_model.predict(X_stress)[0])
        stress_level = stress_mapping.get(predicted_class, "Unknown")
        
        return {
            "stress_class": predicted_class,
            "stress_level": stress_level
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting stress level: {str(e)}")

class SleepLongevityInput(BaseModel):
    age: int
    gender: str
    height: float
    weight: float
    years_poor_sleep: float
    bedtime: str
    wakeup_time: str
    sleep_duration: float
    sleep_quality: str
    exercise_frequency: float
    smoking_status: str
    alcohol_consumption: str
    work_schedule: str
    screen_time: float
    stress_level: float
    existing_conditions: str
    family_heart_disease: bool
    family_diabetes: bool
    family_hypertension: bool

@app.post("/api/predict/sleep_longevity")
async def predict_sleep_longevity(data: SleepLongevityInput):
    try:
        # 1. Calculate BMI
        height_m = data.height / 100.0
        bmi = 0.0
        if height_m > 0:
            bmi = data.weight / (height_m ** 2)
        
        if bmi < 18.5:
            bmi_category = "Underweight"
        elif bmi < 25.0:
            bmi_category = "Normal"
        elif bmi < 30.0:
            bmi_category = "Overweight"
        else:
            bmi_category = "Obese"

        # 2. Evaluate Sleep Deprivation Severity
        if data.sleep_duration < 5:
            sleep_dep_severity = "Severe Sleep Deprivation"
        elif data.sleep_duration < 6.5:
            sleep_dep_severity = "Moderate Sleep Deprivation"
        elif data.sleep_duration < 7.5:
            sleep_dep_severity = "Mild Sleep Deprivation"
        else:
            if data.sleep_quality == "Poor":
                sleep_dep_severity = "Inefficient Sleep / Poor Quality Rest"
            else:
                sleep_dep_severity = "Optimal Sleep Duration"
        
        if data.years_poor_sleep >= 2 and data.sleep_duration < 7.5:
            sleep_dep_severity = f"Chronic {sleep_dep_severity}"

        # 3. Evaluate Lifestyle Risk
        lifestyle_risk_val = 0
        if data.smoking_status == "Heavy smoker":
            lifestyle_risk_val += 30
        elif data.smoking_status == "Light smoker":
            lifestyle_risk_val += 15
        elif data.smoking_status == "Former smoker":
            lifestyle_risk_val += 5
            
        if data.alcohol_consumption == "Heavy":
            lifestyle_risk_val += 20
        elif data.alcohol_consumption == "Moderate":
            lifestyle_risk_val += 10
        elif data.alcohol_consumption == "Occasional":
            lifestyle_risk_val += 3
            
        if data.exercise_frequency < 2:
            lifestyle_risk_val += 15
        elif data.exercise_frequency <= 3:
            lifestyle_risk_val += 5
            
        if data.work_schedule in ["Night Shift", "Rotating Shift"]:
            lifestyle_risk_val += 15
            
        if data.screen_time > 8:
            lifestyle_risk_val += 10
        elif data.screen_time >= 5:
            lifestyle_risk_val += 5
            
        if data.stress_level > 7:
            lifestyle_risk_val += 15
        elif data.stress_level >= 4:
            lifestyle_risk_val += 7
            
        lifestyle_risk_val = min(100, max(0, lifestyle_risk_val))
        if lifestyle_risk_val < 20:
            lifestyle_risk_level = "Low"
        elif lifestyle_risk_val < 45:
            lifestyle_risk_level = "Moderate"
        elif lifestyle_risk_val < 70:
            lifestyle_risk_level = "High"
        else:
            lifestyle_risk_level = "Critical"

        # 4. Generate Sleep Health Score (0-100)
        sleep_health = 100
        if data.sleep_quality == "Poor":
            sleep_health -= 25
        elif data.sleep_quality == "Average":
            sleep_health -= 12
        elif data.sleep_quality == "Excellent":
            sleep_health += 5
            
        if data.sleep_duration < 7:
            sleep_health -= int((7 - data.sleep_duration) * 15)
        elif data.sleep_duration > 9:
            sleep_health -= int((data.sleep_duration - 9) * 10)
            
        try:
            bed_hour = int(data.bedtime.split(":")[0])
            if 1 <= bed_hour <= 5:
                sleep_health -= 10
        except:
            pass
            
        sleep_health -= int(min(20.0, data.years_poor_sleep * 2.5))
        sleep_health = max(10, min(100, sleep_health))

        # 5. Generate Longevity Wellness Score (0-100)
        longevity_wellness = 100
        longevity_wellness -= int((100 - sleep_health) * 0.4)
        
        if bmi >= 30:
            longevity_wellness -= 15
        elif bmi >= 25 or bmi < 18.5:
            longevity_wellness -= 5
            
        if data.smoking_status == "Heavy smoker":
            longevity_wellness -= 15
        elif data.smoking_status == "Light smoker":
            longevity_wellness -= 8
        elif data.smoking_status == "Former smoker":
            longevity_wellness -= 3
            
        if data.alcohol_consumption == "Heavy":
            longevity_wellness -= 12
        elif data.alcohol_consumption == "Moderate":
            longevity_wellness -= 4
            
        if data.exercise_frequency == 0:
            longevity_wellness -= 10
        elif data.exercise_frequency < 2:
            longevity_wellness -= 5
            
        if data.family_heart_disease:
            longevity_wellness -= 8
        if data.family_diabetes:
            longevity_wellness -= 5
        if data.family_hypertension:
            longevity_wellness -= 5
            
        longevity_wellness -= int(data.stress_level * 1.5)
        
        if data.age > 50:
            longevity_wellness -= int(min(10, (data.age - 50) * 0.2))
            
        longevity_wellness = max(15, min(100, longevity_wellness))

        # 6. Biological Risk Level
        if longevity_wellness >= 85 and sleep_health >= 80:
            bio_risk = "Low"
        elif longevity_wellness >= 70 and sleep_health >= 65:
            bio_risk = "Moderate"
        elif longevity_wellness >= 50 and sleep_health >= 45:
            bio_risk = "High"
        else:
            bio_risk = "Critical"

        # 7. Health Risk Predictions
        risks = {}
        
        # Heart Disease
        hd_prob = 10
        if data.family_heart_disease: hd_prob += 25
        if data.age > 45: hd_prob += 15
        if data.sleep_duration < 6: hd_prob += 20
        if data.smoking_status in ["Heavy smoker", "Light smoker"]: hd_prob += 20
        if bmi >= 30: hd_prob += 15
        risks["Heart Disease"] = min(95, hd_prob)
        
        # Hypertension
        ht_prob = 15
        if data.family_hypertension: ht_prob += 25
        if data.sleep_duration < 6: ht_prob += 18
        if data.stress_level > 6: ht_prob += 15
        if data.alcohol_consumption in ["Heavy", "Moderate"]: ht_prob += 12
        if bmi >= 28: ht_prob += 15
        risks["Hypertension"] = min(95, ht_prob)
        
        # Diabetes
        db_prob = 10
        if data.family_diabetes: db_prob += 30
        if bmi >= 30: db_prob += 25
        elif bmi >= 25: db_prob += 10
        if data.exercise_frequency < 2: db_prob += 15
        if data.sleep_duration < 6 or data.sleep_quality == "Poor": db_prob += 12
        risks["Diabetes"] = min(95, db_prob)
        
        # Obesity
        ob_prob = 5
        if bmi >= 30: ob_prob = int(85 + min(14, (bmi - 30) * 2))
        elif bmi >= 25: ob_prob = int(45 + (bmi - 25) * 8)
        else: ob_prob = int(max(2, 5 + (bmi - 18) * 2))
        risks["Obesity"] = min(99, ob_prob)
        
        # Depression
        dp_prob = 12
        if data.sleep_quality == "Poor": dp_prob += 25
        elif data.sleep_quality == "Average": dp_prob += 10
        if data.stress_level > 7: dp_prob += 25
        elif data.stress_level > 4: dp_prob += 10
        if data.sleep_duration < 5.5: dp_prob += 15
        risks["Depression"] = min(90, dp_prob)
        
        # Anxiety
        ax_prob = 15
        if data.stress_level > 6: ax_prob += 30
        elif data.stress_level > 3: ax_prob += 12
        if data.sleep_quality == "Poor": ax_prob += 15
        if data.screen_time > 8: ax_prob += 10
        risks["Anxiety"] = min(92, ax_prob)
        
        # Stroke
        st_prob = 5
        if data.age > 55: st_prob += 15
        if data.family_heart_disease: st_prob += 10
        if data.smoking_status in ["Heavy smoker", "Light smoker"]: st_prob += 20
        if data.sleep_duration < 6: st_prob += 15
        if data.alcohol_consumption == "Heavy": st_prob += 12
        risks["Stroke"] = min(85, st_prob)
        
        # Cognitive Decline
        cd_prob = 8
        if data.age > 60: cd_prob += 20
        elif data.age > 40: cd_prob += 8
        if data.sleep_quality == "Poor" and data.sleep_duration < 6: cd_prob += 20
        if data.screen_time > 8: cd_prob += 10
        risks["Cognitive Decline"] = min(80, cd_prob)
        
        # Memory Problems
        mp_prob = 10
        if data.sleep_quality == "Poor": mp_prob += 25
        if data.sleep_duration < 6: mp_prob += 15
        if data.stress_level > 7: mp_prob += 15
        risks["Memory Problems"] = min(85, mp_prob)
        
        # Hormonal Imbalance
        hi_prob = 12
        if data.work_schedule in ["Night Shift", "Rotating Shift"]: hi_prob += 25
        if data.sleep_duration < 6: hi_prob += 20
        if data.stress_level > 6: hi_prob += 15
        risks["Hormonal Imbalance"] = min(85, hi_prob)
        
        # Immune System Weakness
        is_prob = 15
        if data.sleep_duration < 6: is_prob += 25
        if data.sleep_quality == "Poor": is_prob += 15
        if data.exercise_frequency < 2: is_prob += 12
        risks["Immune System Weakness"] = min(90, is_prob)
        
        # Chronic Fatigue
        cf_prob = 10
        if data.sleep_quality == "Poor": cf_prob += 35
        if data.sleep_duration < 6: cf_prob += 25
        if data.stress_level > 6: cf_prob += 15
        if data.work_schedule == "Night Shift": cf_prob += 10
        risks["Chronic Fatigue"] = min(95, cf_prob)

        # 8. Longevity Details
        if longevity_wellness >= 90:
            health_span_cat = "Elite Health Span (Optimal longevity and cognitive resilience)"
            wellness_rating = "A+"
            habits_impact = "Your sleep and lifestyle habits are highly optimized. This creates an environment for excellent cellular repair, strong autonomic tone, and low systemic inflammation, supporting longevity."
        elif longevity_wellness >= 75:
            health_span_cat = "Robust Health Span (Strong long-term vitality)"
            wellness_rating = "B"
            habits_impact = "Your sleep and lifestyle are generally supportive of health, though small adjustments could enhance cellular resilience and energy levels over the long term."
        elif longevity_wellness >= 55:
            health_span_cat = "Moderate Health Span (Average long-term resilience)"
            wellness_rating = "C"
            habits_impact = "Suboptimal sleep patterns or lifestyle choices are creating mild systemic stress. Addressing sleep debt and scheduling regularity could improve cardiorespiratory stability and metabolic profiles."
        else:
            health_span_cat = "At Risk Health Span (Early onset of chronic age-related stress)"
            wellness_rating = "F"
            habits_impact = "Chronic sleep deprivation, combined with lifestyle or medical risks, places significant stress on metabolic, immune, and cardiovascular systems. Strategic changes are recommended to protect health span."

        future_trends = (
            "If current sleep and stress factors persist, metabolic efficiency and cardiovascular adaptability may decline over the next 10 years. "
            "Conversely, optimizing sleep duration to 7.5+ hours and managing circadian rhythm consistency has been shown to reduce risk indexes of chronic fatigue "
            "and age-related vascular changes."
        )

        # 9. AI Report Generation
        sleep_analysis_report = (
            f"Subject sleeps an average of {data.sleep_duration} hours per night with '{data.sleep_quality}' quality. "
            f"The bedtime is typically around {data.bedtime} and wake-up time is {data.wakeup_time}. "
        )
        if data.sleep_duration < 7:
            sleep_analysis_report += (
                "The analysis indicates significant sleep restriction. Sustained sleep duration below 7 hours "
                "interferes with deep slow-wave sleep, which is critical for physical recovery, growth hormone release, "
                "and muscle tissue repair. Furthermore, it impairs REM sleep, affecting memory consolidation and emotional regulation."
            )
        else:
            sleep_analysis_report += (
                "Sleep duration is within the recommended clinical range. However, if quality is low, "
                "this might indicate frequent micro-arousals, sleep fragmentation, or inadequate deep sleep phases."
            )
            
        if data.years_poor_sleep > 0:
            sleep_analysis_report += f" The history of poor sleep spans {data.years_poor_sleep} years, implying a cumulative sleep debt."

        lifestyle_report = (
            f"Physical exercise occurs {data.exercise_frequency} times per week. "
            f"Work schedule is configured as a {data.work_schedule}. Smoking: {data.smoking_status}; Alcohol: {data.alcohol_consumption}. "
            f"Daily screen time is estimated at {data.screen_time} hours, with a self-assessed stress level of {data.stress_level}/10."
        )
        if data.work_schedule in ["Night Shift", "Rotating Shift"]:
            lifestyle_report += (
                " Operating on shift work disrupts the master circadian pacemaker (suprachiasmatic nucleus), "
                "leading to internal desynchrony, altered cortisol rhythms, and increased cardiorespiratory stress."
            )
        if data.screen_time > 6:
            lifestyle_report += (
                " High screen exposure, especially close to bedtime, exposes the eyes to blue light, suppressing "
                "natural melatonin synthesis and delaying sleep onset."
            )

        risk_report = (
            f"The computed Biological Risk Level is {bio_risk} (Longevity Score: {longevity_wellness}/100, Sleep Score: {sleep_health}/100). "
            f"Elevated probabilities are observed for "
        )
        elevated_risks = [k for k, v in risks.items() if v > 40]
        if elevated_risks:
            risk_report += ", ".join(elevated_risks) + ". "
        else:
            risk_report += "none of the major assessed chronic conditions at this stage. "
            
        risk_report += (
            "Family history of chronic diseases and elevated lifestyle risks directly shift these probability baselines. "
            "Chronic sleep deficiency elevates sympathetic nervous system activity, which contributes to arterial stiffness and blood pressure changes."
        )

        future_concerns_report = (
            "Unresolved sleep deficiency is associated with the long-term accumulation of amyloid-beta plaques in the brain, "
            "promoting cognitive aging. Furthermore, chronic inflammation and insulin resistance may compound over 5 to 15 years, "
            "elevating vascular risk and compromising immune defense mechanisms."
        )

        improvements = [
            "Establish a highly consistent sleep schedule, maintaining the same bedtime and wake-up time even on weekends.",
            "Limit exposure to bright screens (phones, computers, televisions) for at least 60 minutes before bedtime.",
            f"Increase weekly physical exercise to at least 3-4 days (current: {data.exercise_frequency} days/week) to boost slow-wave sleep.",
            "Practice relaxation techniques (deep breathing, progressive muscle relaxation) to decrease pre-sleep arousal and lower stress levels."
        ]
        if data.smoking_status in ["Heavy smoker", "Light smoker"]:
            improvements.append("Seek smoking cessation support, as nicotine is a stimulant that disrupts sleep architecture.")
        if data.alcohol_consumption == "Heavy":
            improvements.append("Reduce alcohol consumption, particularly in the evening, as alcohol disrupts REM sleep sleep cycles.")
        if data.work_schedule in ["Night Shift", "Rotating Shift"]:
            improvements.append("Use blackout curtains and white noise machines to optimize daytime sleep conditions during night shifts.")

        rec_bedtime = "22:30"
        rec_wakeup = "06:30"
        if data.work_schedule == "Day Shift":
            rec_bedtime = "22:30"
            rec_wakeup = "06:30"
        elif data.work_schedule == "Night Shift":
            rec_bedtime = "08:30"
            rec_wakeup = "16:30"
        else:
            rec_bedtime = "Variable (aim for a consistent 8-hour window)"
            rec_wakeup = "Variable"

        rec_sleep = (
            f"Target a consistent sleep window: {rec_bedtime} to {rec_wakeup} (8.0 hours). "
            f"Keep bedroom temperature cool (around 18-20°C). Avoid large meals and caffeine within 6 hours of sleep."
        )

        routine = [
            "07:00 AM - Wake up and get 10-15 minutes of direct sunlight exposure to anchor circadian rhythms.",
            "08:00 AM - Hydrate and consume a balanced high-protein breakfast.",
            "01:00 PM - Final caffeine consumption cutoff for the day.",
            "05:00 PM - Moderate cardiovascular exercise or resistance training (at least 30 mins).",
            "07:30 PM - Light dinner, avoiding heavy, spicy, or high-sugar foods.",
            "09:00 PM - Enable warm lighting, disconnect from work, and begin blue-light restriction.",
            "10:00 PM - Wind down with reading, journaling, or stretching; avoid stimulating topics.",
            "10:30 PM - Bedroom lights out; maintain a cool, dark, and silent environment."
        ]
        if data.work_schedule == "Night Shift":
            routine = [
                "04:30 PM - Wake up and expose eyes to bright indoor light to signal alertness.",
                "05:30 PM - Consume a balanced meal (breakfast-equivalent).",
                "09:00 PM - Start work shift; keep work environment well-lit.",
                "01:00 AM - Midnight meal / high-protein snack; maintain hydration.",
                "07:00 AM - End shift; wear blue-blocking sunglasses on the commute home.",
                "07:30 AM - Light pre-sleep snack, avoid caffeine or heavy meals.",
                "08:00 AM - Wind down in a dim environment; block out all screens.",
                "08:30 AM - Sleep in a bedroom blacked out with heavy curtains; use earplugs/white noise."
            ]

        return {
            "bmi": round(bmi, 2),
            "bmi_category": bmi_category,
            "sleep_deprivation_severity": sleep_dep_severity,
            "lifestyle_risk_score": lifestyle_risk_val,
            "lifestyle_risk_level": lifestyle_risk_level,
            "sleep_health_score": sleep_health,
            "longevity_wellness_score": longevity_wellness,
            "biological_risk_level": bio_risk,
            "health_risks": risks,
            "longevity_details": {
                "health_span_category": health_span_cat,
                "overall_wellness_rating": wellness_rating,
                "expected_impact": habits_impact,
                "future_risk_trends": future_trends
            },
            "ai_report": {
                "sleep_analysis": sleep_analysis_report,
                "lifestyle_analysis": lifestyle_report,
                "health_risk_analysis": risk_report,
                "future_concerns": future_concerns_report,
                "recommended_improvements": improvements,
                "personalized_sleep_recommendations": rec_sleep,
                "suggested_daily_routine": routine
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating sleep health analysis: {str(e)}")

# Mount the static frontend files
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
else:
    print("WARNING: 'static' directory not found. Frontend will not be served.")

if __name__ == "__main__":
    import uvicorn
    print("Starting server on http://localhost:8000")
    uvicorn.run("app_api:app", host="0.0.0.0", port=8000, reload=True)
