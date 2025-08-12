import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional
import google.generativeai as genai
import os

# Page configuration
st.set_page_config(
    page_title="AI Workout Builder",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .workout-card {
        background: #cff00c;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 10px 0;
        border-left: 5px solid #667eea;
    }
    .exercise-item {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 5px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102,126,234,0.3);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'workout_plan' not in st.session_state:
        st.session_state.workout_plan = None
    if 'exercises_db' not in st.session_state:
        st.session_state.exercises_db = []
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {}

init_session_state()

# API Configuration
class ExerciseAPI:
    """Handler for exercise data from external APIs"""
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def fetch_exercises_ninja_api(muscle_group: str = None) -> List[Dict]:
        """Fetch exercises from API-Ninjas Exercise API"""
        try:
            api_key = st.secrets.get("NINJA_API_KEY", "your_ninja_api_key_here")
            url = "https://api.api-ninjas.com/v1/exercises"
            headers = {"X-Api-Key": api_key}
            
            params = {}
            if muscle_group:
                params['muscle'] = muscle_group.lower()
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.warning(f"API returned status code: {response.status_code}")
                return ExerciseAPI.get_fallback_exercises(muscle_group)
        except Exception as e:
            st.warning(f"API error: {str(e)}")
            return ExerciseAPI.get_fallback_exercises(muscle_group)
    
    @staticmethod
    def get_fallback_exercises(muscle_group: str = None) -> List[Dict]:
        """Fallback exercise database when API is unavailable"""
        exercises = [
            {
                "name": "Push-ups",
                "type": "strength",
                "muscle": "chest",
                "equipment": "body_only",
                "difficulty": "beginner",
                "instructions": "Start in plank position. Lower body until chest nearly touches floor. Push back up to start position."
            },
            {
                "name": "Squats",
                "type": "strength", 
                "muscle": "quadriceps",
                "equipment": "body_only",
                "difficulty": "beginner",
                "instructions": "Stand with feet shoulder-width apart. Lower body by bending knees and hips. Return to starting position."
            },
            {
                "name": "Plank",
                "type": "strength",
                "muscle": "abdominals",
                "equipment": "body_only", 
                "difficulty": "beginner",
                "instructions": "Hold a push-up position with forearms on ground. Keep body straight from head to heels."
            },
            {
                "name": "Lunges",
                "type": "strength",
                "muscle": "quadriceps",
                "equipment": "body_only",
                "difficulty": "intermediate",
                "instructions": "Step forward with one leg, lowering hips until both knees are bent at 90 degrees. Return to start."
            },
            {
                "name": "Burpees",
                "type": "cardio",
                "muscle": "full_body",
                "equipment": "body_only",
                "difficulty": "intermediate",
                "instructions": "From standing, drop to squat, jump back to plank, do push-up, jump forward to squat, jump up."
            },
            {
                "name": "Mountain Climbers",
                "type": "cardio",
                "muscle": "abdominals",
                "equipment": "body_only",
                "difficulty": "intermediate", 
                "instructions": "Start in plank position. Alternate bringing knees to chest in running motion."
            },
            {
                "name": "Deadlifts",
                "type": "strength",
                "muscle": "hamstrings",
                "equipment": "barbell",
                "difficulty": "intermediate",
                "instructions": "Stand with barbell at feet. Bend at hips and knees to lower bar. Stand up by extending hips and knees."
            },
            {
                "name": "Pull-ups",
                "type": "strength",
                "muscle": "lats",
                "equipment": "pull_up_bar",
                "difficulty": "intermediate",
                "instructions": "Hang from pull-up bar with palms facing away. Pull body up until chin clears bar."
            },
            {
                "name": "Bicep Curls",
                "type": "strength",
                "muscle": "biceps",
                "equipment": "dumbbell",
                "difficulty": "beginner",
                "instructions": "Hold dumbbells at sides. Curl weights up by flexing biceps. Lower slowly to start."
            },
            {
                "name": "Tricep Dips",
                "type": "strength",
                "muscle": "triceps",
                "equipment": "body_only",
                "difficulty": "beginner",
                "instructions": "Sit on chair edge, hands gripping seat. Lower body by bending arms, then push back up."
            }
        ]
        
        if muscle_group:
            exercises = [ex for ex in exercises if muscle_group.lower() in ex['muscle'].lower()]
        
        return exercises

class WorkoutAI:
    """AI-powered workout plan generator using Google Gemini"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or st.secrets.get("GEMINI_API_KEY", "")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None
    
    def generate_workout_plan(self, user_profile: Dict, exercises: List[Dict]) -> Dict:
        """Generate AI-powered workout plan"""
        if not self.model:
            return self._generate_fallback_plan(user_profile, exercises)
        
        try:
            # Prepare exercise data for AI
            exercise_summary = "\n".join([
                f"- {ex['name']}: {ex['type']} exercise for {ex['muscle']}, "
                f"difficulty: {ex['difficulty']}, equipment: {ex['equipment']}"
                for ex in exercises[:20]  # Limit to avoid token limits
            ])
            
            prompt = f"""
            Create a personalized weekly workout plan based on:
            
            USER PROFILE:
            - Goal: {user_profile.get('goal', 'general fitness')}
            - Experience Level: {user_profile.get('experience', 'beginner')}
            - Days per Week: {user_profile.get('days_per_week', 3)}
            - Session Duration: {user_profile.get('duration', 45)} minutes
            - Equipment Available: {user_profile.get('equipment', 'bodyweight only')}
            - Focus Areas: {user_profile.get('focus_areas', ['full body'])}
            
            AVAILABLE EXERCISES:
            {exercise_summary}
            
            Please create a structured weekly plan with:
            1. Day-by-day workout schedule
            2. Specific exercises with sets, reps, and rest periods
            3. Progression recommendations
            4. Tips for success
            
            Format as JSON with this structure:
            {{
                "plan_name": "Personalized Plan Name",
                "overview": "Brief description",
                "weekly_schedule": {{
                    "day_1": {{
                        "focus": "muscle group focus",
                        "exercises": [
                            {{
                                "name": "exercise name",
                                "sets": 3,
                                "reps": "12-15",
                                "rest": "60 seconds",
                                "notes": "form tips"
                            }}
                        ]
                    }}
                }},
                "progression_tips": ["tip1", "tip2"],
                "success_tips": ["tip1", "tip2"]
            }}
            
            Return ONLY the JSON response, no additional text or formatting.
            """
            
            # Generate response with Gemini
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=2048,
                    candidate_count=1
                )
            )
            
            # Parse JSON response
            plan_text = response.text.strip()
            
            # Clean up response text (remove markdown formatting if present)
            if plan_text.startswith('```json'):
                plan_text = plan_text.replace('```json', '').replace('```', '').strip()
            elif plan_text.startswith('```'):
                plan_text = plan_text.replace('```', '').strip()
            
            # Extract JSON from response
            start_idx = plan_text.find('{')
            end_idx = plan_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = plan_text[start_idx:end_idx]
                plan = json.loads(json_str)
                return plan
            else:
                # If JSON parsing fails, try to parse the entire response
                try:
                    plan = json.loads(plan_text)
                    return plan
                except:
                    return self._generate_fallback_plan(user_profile, exercises)
                
        except Exception as e:
            st.warning(f"AI generation error: {str(e)}")
            return self._generate_fallback_plan(user_profile, exercises)
    
    def _generate_fallback_plan(self, user_profile: Dict, exercises: List[Dict]) -> Dict:
        """Generate a basic workout plan when AI is unavailable"""
        days_per_week = user_profile.get('days_per_week', 3)
        experience = user_profile.get('experience', 'beginner')
        goal = user_profile.get('goal', 'general_fitness')
        
        # Filter exercises by difficulty
        suitable_exercises = [
            ex for ex in exercises 
            if ex['difficulty'] == experience or experience == 'advanced'
        ]
        
        if not suitable_exercises:
            suitable_exercises = exercises
        
        # Basic plan structure
        plan = {
            "plan_name": f"Custom {experience.title()} {goal.replace('_', ' ').title()} Plan",
            "overview": f"A {days_per_week}-day per week workout plan focused on {goal.replace('_', ' ')}",
            "weekly_schedule": {},
            "progression_tips": [
                "Increase weight/reps by 5-10% when you can complete all sets easily",
                "Focus on proper form before increasing intensity",
                "Rest at least one day between intense workouts"
            ],
            "success_tips": [
                "Stay consistent with your schedule",
                "Track your progress",
                "Listen to your body and rest when needed",
                "Stay hydrated and eat well"
            ]
        }
        
        # Generate daily workouts
        for day in range(1, days_per_week + 1):
            selected_exercises = suitable_exercises[:4]  # Select first 4 exercises
            
            if experience == 'beginner':
                sets, reps, rest = 2, "8-12", "90 seconds"
            elif experience == 'intermediate':
                sets, reps, rest = 3, "10-15", "60 seconds"
            else:
                sets, reps, rest = 3, "12-20", "45 seconds"
            
            plan["weekly_schedule"][f"day_{day}"] = {
                "focus": "Full Body" if day <= 3 else "Active Recovery",
                "exercises": [
                    {
                        "name": ex['name'],
                        "sets": sets,
                        "reps": reps,
                        "rest": rest,
                        "notes": "Focus on proper form"
                    }
                    for ex in selected_exercises
                ]
            }
        
        return plan

# Main App Interface
def main():
    st.title(" AI-Powered Workout Routine Builder")
    st.markdown("### Create personalized workout plans powered by AI and real exercise data")
    
    # Sidebar for user input
    with st.sidebar:
        st.header(" Your Fitness Profile")
        
        # User profile inputs
        goal = st.selectbox(
            "Primary Goal",
            ["general_fitness", "weight_loss", "muscle_gain", "strength", "endurance", "flexibility"],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        experience = st.selectbox(
            "Experience Level",
            ["beginner", "intermediate", "advanced"]
        )
        
        days_per_week = st.slider("Workout Days per Week", 2, 7, 4)
        
        duration = st.slider("Session Duration (minutes)", 15, 120, 45)
        
        equipment = st.multiselect(
            "Available Equipment",
            ["body_only", "dumbbells", "barbell", "resistance_bands", "pull_up_bar", "kettlebell", "gym_access"],
            default=["body_only"],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        focus_areas = st.multiselect(
            "Focus Areas",
            ["chest", "back", "shoulders", "arms", "legs", "core", "cardio", "full_body"],
            default=["full_body"]
        )
        

    
    # Store user profile
    st.session_state.user_profile = {
        'goal': goal,
        'experience': experience,
        'days_per_week': days_per_week,
        'duration': duration,
        'equipment': equipment,
        'focus_areas': focus_areas
    }
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("###  Your Profile")
        st.markdown(f"""
        <div class="metric-card">
            <h4> {goal.replace('_', ' ').title()}</h4>
            <p><strong>Level:</strong> {experience.title()}</p>
            <p><strong>Frequency:</strong> {days_per_week} days/week</p>
            <p><strong>Duration:</strong> {duration} minutes</p>
            <p><strong>Equipment:</strong> {', '.join([e.replace('_', ' ').title() for e in equipment])}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col1:
        # Generate workout plan button
        if st.button(" Generate My Workout Plan", type="primary"):
            with st.spinner("Creating your personalized workout plan..."):
                # Fetch exercises
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Fetching exercise database...")
                progress_bar.progress(25)
                
                # Get exercises for focus areas
                all_exercises = []
                for focus in focus_areas:
                    exercises = ExerciseAPI.fetch_exercises_ninja_api(focus)
                    all_exercises.extend(exercises)
                
                # Remove duplicates
                seen_names = set()
                unique_exercises = []
                for ex in all_exercises:
                    if ex['name'] not in seen_names:
                        unique_exercises.append(ex)
                        seen_names.add(ex['name'])
                
                st.session_state.exercises_db = unique_exercises
                
                status_text.text("Generating AI-powered workout plan...")
                progress_bar.progress(75)
                
                # Generate workout plan
                workout_ai = WorkoutAI()
                plan = workout_ai.generate_workout_plan(st.session_state.user_profile, unique_exercises)
                st.session_state.workout_plan = plan
                
                status_text.text("Workout plan generated successfully!")
                progress_bar.progress(100)
                time.sleep(1)
                
                progress_bar.empty()
                status_text.empty()
    
    # Display workout plan
    if st.session_state.workout_plan:
        st.markdown("---")
        plan = st.session_state.workout_plan
        
        st.markdown(f"##  {plan['plan_name']}")
        st.markdown(f"**Overview:** {plan['overview']}")
        
        # Weekly schedule tabs
        schedule = plan['weekly_schedule']
        day_tabs = st.tabs([f"Day {i}" for i in range(1, len(schedule) + 1)])
        
        for i, (day_key, day_data) in enumerate(schedule.items()):
            with day_tabs[i]:
                st.markdown(f"###  Focus: {day_data['focus']}")
                
                for j, exercise in enumerate(day_data['exercises']):
                    st.markdown(f"""
                    <div class="exercise-item">
                        <h4>{j+1}. {exercise['name']}</h4>
                        <div style="display: flex; gap: 20px; margin-top: 10px;">
                            <span><strong>Sets:</strong> {exercise['sets']}</span>
                            <span><strong>Reps:</strong> {exercise['reps']}</span>
                            <span><strong>Rest:</strong> {exercise['rest']}</span>
                        </div>
                        <p style="margin-top: 10px; font-style: italic;">{exercise.get('notes', '')}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Tips and progression
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("###  Progression Tips")
            for tip in plan.get('progression_tips', []):
                st.markdown(f"• {tip}")
        
        with col2:
            st.markdown("###  Success Tips")
            for tip in plan.get('success_tips', []):
                st.markdown(f"• {tip}")
        
        # Export options
        st.markdown("###  Export Your Plan")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(" Download as JSON"):
                json_str = json.dumps(plan, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"workout_plan_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
        

        
        with col2:
            if st.button(" Generate New Plan"):
                st.session_state.workout_plan = None
                st.rerun()
    


if __name__ == "__main__":
    main()
