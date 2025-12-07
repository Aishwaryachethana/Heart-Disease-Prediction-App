from flask import Flask, request, jsonify
from flask_cors import CORS
from io import StringIO
import base64
import pandas as pd

from . import nlp as nlp_mod
from . import stats as stats_mod
from . import charts as charts_mod
from . import suggestions as sugg_mod
from . import cleaning as clean_mod
from . import ml as ml_mod

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)


@app.route('/')
def index():
    # serve the frontend index.html
    return app.send_static_file('index.html')


def load_csv_from_text(csv_text):
    return pd.read_csv(StringIO(csv_text))


@app.route('/api/process', methods=['POST'])
def process_prompt():
    data = request.get_json()
    prompt = data.get('prompt', '')
    csv_text = data.get('csv_text')

    # handle follow-up answers to clarifying questions
    followup_for = data.get('followup_for')
    followup_answer = data.get('followup_answer')

    if not csv_text:
        return jsonify({'error': 'No CSV content provided'}), 400

    try:
        df = load_csv_from_text(csv_text)
    except Exception as e:
        return jsonify({'error': f'Failed to parse CSV: {e}'}), 400

    # If this request contains a follow-up answer (reply to a clarification),
    # treat the follow-up differently: parse the follow-up and apply it to the
    # original pending intent (e.g., chart columns).
    if followup_for and followup_answer:
        # parse followup to extract columns or target
        f_intent, f_entities = nlp_mod.parse_prompt(followup_answer)
        # for now we only support followups for chart and ml target
        if followup_for == 'chart':
            cols = f_entities.get('columns', [])
            mapped = nlp_mod.match_columns(df.columns.tolist(), cols)
            x = mapped[0] if len(mapped) > 0 else None
            y = mapped[1] if len(mapped) > 1 else None
            if not x or not y:
                return jsonify({'clarify': True, 'question': 'Which columns should be X and Y?', 'expected': 'columns', 'intent': 'chart'}), 200
            img_b64 = charts_mod.generate_chart(df, x, y, kind=f_entities.get('chart_type', 'bar'))
            return jsonify({'intent': 'chart', 'chart_base64': img_b64})
        if followup_for == 'ml':
            # attempt to use followup as target column name
            target = f_entities.get('target') or (f_entities.get('columns') and f_entities.get('columns')[0])
            if target:
                mapped = nlp_mod.match_columns(df.columns.tolist(), [target])
                if mapped:
                    model_info = ml_mod.train_simple_model(df, mapped[0])
                    return jsonify({'intent': 'ml', 'ml': model_info})
            return jsonify({'clarify': True, 'question': 'Which column is the target?', 'expected': 'target', 'intent': 'ml'}), 200

    # basic NLP parsing
    intent, entities = nlp_mod.parse_prompt(prompt)

    # fuzzy match column names
    mapped = nlp_mod.match_columns(df.columns.tolist(), entities.get('columns', []))

    response = {'intent': intent, 'mapped_columns': mapped}

    # handle some intents
    if intent == 'descriptive_stats':
        cols = mapped if mapped else df.select_dtypes(include='number').columns.tolist()
        table = stats_mod.descriptive_stats(df, cols)
        response.update({'table': table})
        return jsonify(response)

    if intent == 'chart':
        # expect x and y
        x = mapped[0] if len(mapped) > 0 else None
        y = mapped[1] if len(mapped) > 1 else None
        if not x or not y:
            return jsonify({'clarify': True, 'question': 'Which columns should be X and Y?'}), 200
        img_b64 = charts_mod.generate_chart(df, x, y, kind=entities.get('chart_type', 'line'))
        response.update({'chart_base64': img_b64})
        return jsonify(response)

    if intent == 'cleaning':
        cleaned = clean_mod.suggest_cleaning(df)
        response.update({'cleaning_suggestions': cleaned})
        return jsonify(response)

    if intent == 'suggestions':
        s = sugg_mod.generate_suggestions(df)
        response.update({'suggestions': s})
        return jsonify(response)

    if intent == 'predict' or intent == 'ml':
        # naive: try to train a simple regressor if numeric target given
        if 'target' in entities and entities['target']:
            target = nlp_mod.match_columns(df.columns.tolist(), [entities['target']])
            if target:
                model_info = ml_mod.train_simple_model(df, target[0])
                response.update({'ml': model_info})
                return jsonify(response)
            else:
                return jsonify({'clarify': True, 'question': 'Which column is the target?'}), 200

    # fallback
    return jsonify({'reply': "I understood: %s. I can run descriptive stats, charts, cleaning, suggestions, or ML. Try: 'show summary' or 'plot x vs y'." % intent})


if __name__ == '__main__':
    app.run(debug=True, port=8787)
