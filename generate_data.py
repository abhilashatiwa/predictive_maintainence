import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Parameters
num_samples = 1000
start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 12, 31)

# Generate random data
np.random.seed(42)
machine_ids = np.random.randint(1, 100, num_samples)
maintenance_dates = [start_date + timedelta(days=np.random.randint(0, (end_date - start_date).days)) for _ in range(num_samples)]
operating_hours = np.random.randint(0, 10000, num_samples)
temperatures = np.random.uniform(50, 100, num_samples)
vibration_levels = np.random.uniform(0, 10, num_samples)
ages = np.random.randint(1, 20, num_samples)
failures = np.random.randint(0, 2, num_samples)

# Create DataFrame
data = pd.DataFrame({
    'machine_id': machine_ids,
    'maintenance_date': maintenance_dates,
    'operating_hours': operating_hours,
    'temperature': temperatures,
    'vibration_level': vibration_levels,
    'age': ages,
    'failure': failures
})

# Save to CSV
data.to_csv('history_data.csv', index=False)

print("Generated history_data.csv with 1000 samples.")
