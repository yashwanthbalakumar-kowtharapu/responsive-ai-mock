from flask import Flask, jsonify, request, g
import google.generativeai as genai
from functions.question_generation import generate_questions
from functions.emotion_analysis import analyze_fun
from functions.review_generation import gen_review
from flask_cors import CORS, cross_origin
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor
import queue
import json

app = Flask(__name__)
CORS(app)
executor = ThreadPoolExecutor(max_workers=3)
emotion_queue = queue.Queue()

# Configure OpenAI or Google Generative AI
load_dotenv()
gemini_api_key = os.getenv('GEMINI_API_KEY3')
genai.configure(api_key=gemini_api_key)

model = genai.GenerativeModel(model_name="gemini-1.5-flash")

@app.before_request
def before_request():
    g.model = model

@app.route("/")
def home():
    return "Welcome to Mock-Interview-System/Server", 200

@app.errorhandler(404)
def page_not_found(e):
    return jsonify({"status": 404, "message": "Not Found"}), 404

@app.route('/api/get-questions', methods=['POST'])
def getQuestion():
    try:
        data = request.get_json()
        print("Received data for question generation:", data)
        company = data['company']
        job_role = data['job_role']
        experience_lvl = data['experience_lvl']

        # Generate questions
        questions = generate_questions(job_role, experience_lvl, company)
        print("Generated Questions:", questions)

        if not isinstance(questions, list) or not questions:
            print("Failed to generate questions. Response:", questions)
            return jsonify({'errorMsg': 'Failed to generate questions'}), 400

        return jsonify({'job_role': job_role, 'exp_level': experience_lvl, 'qtns': questions}), 200
    except Exception as e:
        print(f"Error occurred while generating question: {e}")
        return jsonify({'errorMsg': "Something went wrong"}), 400

@app.route('/generate_ideal_answer', methods=['POST'])
def generate_ideal_answer():
    try:
        data = request.get_json()
        question = data.get('question')

        # Generate ideal answer
        ideal_answer = f"This is a sample ideal answer for the question: {question}"

        return jsonify({'answer': ideal_answer}), 200
    except Exception as e:
        print(f"Error occurred while generating ideal answer: {e}")
        return jsonify({'errorMsg': "Something went wrong"}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
