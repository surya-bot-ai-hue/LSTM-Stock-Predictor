{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "79946840",
   "metadata": {},
   "source": [
    "# Stock Price Prediction with LSTM\n",
    "\n",
    "This notebook performs stock price prediction using an LSTM model. It includes animated and static visualizations of predicted vs actual stock prices. You can either use a Yahoo Finance dataset or upload your own CSV file."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "777cf1c7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 1: Install Required Libraries\n",
    "!pip install plotly openpyxl yfinance\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "51ce92a7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 2: Import Libraries\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import plotly.graph_objects as go\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import LSTM, Dense\n",
    "from google.colab import files\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "837daa57",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 3: Upload Yahoo Finance CSV file\n",
    "uploaded = files.upload()\n",
    "file_path = list(uploaded.keys())[0]\n",
    "df = pd.read_csv(file_path)\n",
    "df.columns = [col.strip() for col in df.columns]\n",
    "df['Date'] = pd.to_datetime(df['Date'])\n",
    "df.sort_values('Date', inplace=True)\n",
    "df.reset_index(drop=True, inplace=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "54497e89",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 4: Preprocess Data (Normalize and Sequence)\n",
    "scaler = MinMaxScaler()\n",
    "scaled = scaler.fit_transform(df[['Close']])\n",
    "\n",
    "X, y = [], []\n",
    "seq_len = 60\n",
    "for i in range(seq_len, len(scaled)):\n",
    "    X.append(scaled[i - seq_len:i])\n",
    "    y.append(scaled[i])\n",
    "\n",
    "X, y = np.array(X), np.array(y)\n",
    "X = X.reshape((X.shape[0], X.shape[1], 1))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1ff034e5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 5: Build and Train LSTM Model\n",
    "model = Sequential([\n",
    "    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),\n",
    "    LSTM(50),\n",
    "    Dense(1)\n",
    "])\n",
    "model.compile(optimizer='adam', loss='mse')\n",
    "model.fit(X, y, epochs=1, batch_size=32, verbose=0)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b16e9eaa",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 6: Make Predictions and Inverse Transform\n",
    "y_pred = model.predict(X, verbose=0)\n",
    "y_pred_inv = scaler.inverse_transform(y_pred)\n",
    "y_actual_inv = scaler.inverse_transform(y)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2fe33e35",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 7: Animated Visualization (Plotly)\n",
    "def animate_line(line_data, title):\n",
    "    fig = go.Figure(\n",
    "        data=[go.Scatter(x=[], y=[], mode='lines', line=dict(color='blue'))],\n",
    "        layout=go.Layout(\n",
    "            title=title,\n",
    "            xaxis=dict(range=[0, len(line_data)]),\n",
    "            yaxis=dict(range=[min(line_data) - 100, max(line_data) + 100]),\n",
    "            updatemenus=[dict(type='buttons', showactive=False, buttons=[\n",
    "                dict(label='Play', method='animate', args=[None, {'frame': {'duration': 30, 'redraw': True}, 'fromcurrent': True}])\n",
    "            ])]\n",
    "        ),\n",
    "        frames=[go.Frame(data=[go.Scatter(x=list(range(k + 1)), y=line_data[:k + 1])]) for k in range(1, len(line_data))]\n",
    "    )\n",
    "    fig.show()\n",
    "\n",
    "# Animate actual and predicted stock prices\n",
    "animate_line(y_actual_inv.flatten(), '📈 Animated Actual Stock Prices')\n",
    "animate_line(y_pred_inv.flatten(), '🤖 Animated Predicted Stock Prices')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "89bcd757",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 8: Static Visualization (Matplotlib)\n",
    "plt.figure(figsize=(12, 6))\n",
    "plt.plot(range(seq_len, seq_len + len(y_actual_inv)), y_actual_inv, label='Actual', color='blue')\n",
    "plt.plot(range(seq_len, seq_len + len(y_pred_inv)), y_pred_inv, label='Predicted', color='orange')\n",
    "plt.title('Actual vs Predicted Stock Prices')\n",
    "plt.xlabel('Days')\n",
    "plt.ylabel('Stock Price')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}
