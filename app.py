import pickle
from flask import Flask, render_template_string, request
import numpy as np

app = Flask(__name__)

# Load the trained Linear Regression model[cite: 1]
with open('linear.pkl', 'rb') as f:
  model = pickle.load(f)


def format_inr(amount):
  """Formats a number into Indian Rupee style (lakhs, crores)."""
  if amount < 0:
    return f"-₹{format_inr(abs(amount))}"

  # Round to nearest integer for clean display
  s = str(int(round(amount)))
  if len(s) <= 3:
    return f"₹{s}"

  # Extract last 3 digits
  last_three = s[-3:]
  other_digits = s[:-3]

  # Format remaining digits in groups of 2
  res = []
  while len(other_digits) > 0:
    res.insert(0, other_digits[-2:])
    other_digits = other_digits[:-2]

  return f"₹{','.join(res)},{last_three}"


# HTML template with embedded modern CSS styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real Estate Price Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg-gradient: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
            --card-bg: #ffffff;
            --text-main: #1f2937;
            --text-muted: #6b7280;
            --border: #d1d5db;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body {
            background: var(--bg-gradient);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
            color: var(--text-main);
        }
        .container {
            background: var(--card-bg);
            width: 100%;
            max-width: 650px;
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        }
        h1 {
            text-align: center;
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 8px;
            font-size: 24px;
        }
        .subtitle {
            text-align: center;
            color: var(--text-muted);
            font-size: 14px;
            margin-bottom: 30px;
        }
        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        .form-group.full-width {
            grid-column: span 2;
        }
        label {
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 6px;
            color: var(--text-main);
        }
        input {
            padding: 12px 14px;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-size: 14px;
            transition: all 0.2s ease;
            outline: none;
        }
        input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
        }
        button {
            grid-column: span 2;
            background-color: var(--primary);
            color: white;
            border: none;
            padding: 14px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s ease, transform 0.1s ease;
            margin-top: 10px;
        }
        button:hover {
            background-color: var(--primary-hover);
        }
        button:active {
            transform: scale(0.98);
        }
        .result-card {
            margin-top: 25px;
            background: #eef2ff;
            border: 1px solid #c7d2fe;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }
        .result-card h3 {
            color: #3730a3;
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 5px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .price {
            font-size: 28px;
            font-weight: 700;
            color: #312e81;
        }
        @media (max-width: 600px) {
            .form-grid { grid-template-columns: 1fr; }
            .form-group.full-width { grid-column: span 1; }
            button { grid-column: span 1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏡 Property Value Predictor</h1>
        <p class="subtitle">Enter the property specifications to estimate market price.</p>
        
        <form method="POST">
            <div class="form-grid">
                <div class="form-group">
                    <label for="Square_Footage">Square Footage</label>
                    <input type="number" step="any" id="Square_Footage" name="Square_Footage" required placeholder="e.g. 2100">
                </div>
                <div class="form-group">
                    <label for="Num_Bedrooms">Number of Bedrooms</label>
                    <input type="number" id="Num_Bedrooms" name="Num_Bedrooms" required placeholder="e.g. 3">
                </div>
                <div class="form-group">
                    <label for="Num_Bathrooms">Number of Bathrooms</label>
                    <input type="number" step="any" id="Num_Bathrooms" name="Num_Bathrooms" required placeholder="e.g. 2">
                </div>
                <div class="form-group">
                    <label for="Year_Built">Year Built</label>
                    <input type="number" id="Year_Built" name="Year_Built" required placeholder="e.g. 2012">
                </div>
                <div class="form-group">
                    <label for="Lot_Size">Lot Size</label>
                    <input type="number" step="any" id="Lot_Size" name="Lot_Size" required placeholder="e.g. 4500">
                </div>
                <div class="form-group">
                    <label for="Garage_Size">Garage Size (Cars)</label>
                    <input type="number" id="Garage_Size" name="Garage_Size" required placeholder="e.g. 2">
                </div>
                <div class="form-group full-width">
                    <label for="Neighborhood_Quality">Neighborhood Quality Score</label>
                    <input type="number" step="any" id="Neighborhood_Quality" name="Neighborhood_Quality" required placeholder="e.g. 8">
                </div>
                <button type="submit">Calculate Estimated Price</button>
            </div>
        </form>

        {% if formatted_price is not none %}
        <div class="result-card">
            <h3>Estimated Property Price</h3>
            <div class="price">{{ formatted_price }}</div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


@app.route('/', methods=['GET', 'POST'])
def index():
  formatted_price = None
  if request.method == 'POST':
    try:
      # Capture and cast form inputs to match model expectations[cite: 1]
      features = np.array([[
          float(request.form['Square_Footage']),
          float(request.form['Num_Bedrooms']),
          float(request.form['Num_Bathrooms']),
          float(request.form['Year_Built']),
          float(request.form['Lot_Size']),
          float(request.form['Garage_Size']),
          float(request.form['Neighborhood_Quality']),
      ]])

      # Predict raw price value from the model
      raw_prediction = model.predict(features)[0]

      # Format prediction into Indian Rupees layout
      formatted_price = format_inr(raw_prediction)
    except Exception as e:
      formatted_price = None

  return render_template_string(HTML_TEMPLATE, formatted_price=formatted_price)


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)
