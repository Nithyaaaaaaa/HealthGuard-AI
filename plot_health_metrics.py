
import pandas as pd
import matplotlib.pyplot as plt

# Load the time-series vitals data
df = pd.read_csv("vitals_time_series.csv", parse_dates=['timestamp'])

# Plot example patient
example_patient = df[df['patient_id'] == 'P001']

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(example_patient['timestamp'], example_patient['heart_rate'], label='Heart Rate')
plt.ylabel('BPM')
plt.title('Heart Rate')

plt.subplot(2, 2, 2)
plt.plot(example_patient['timestamp'], example_patient['systolic_bp'], label='Systolic BP')
plt.plot(example_patient['timestamp'], example_patient['diastolic_bp'], label='Diastolic BP')
plt.ylabel('mmHg')
plt.title('Blood Pressure')

plt.subplot(2, 2, 3)
plt.plot(example_patient['timestamp'], example_patient['temperature'], label='Temperature', color='orange')
plt.ylabel('°C')
plt.title('Body Temperature')

plt.subplot(2, 2, 4)
plt.plot(example_patient['timestamp'], example_patient['spo2'], label='SpO₂', color='green')
plt.ylabel('%')
plt.title('Oxygen Saturation')

plt.tight_layout()
plt.show()
