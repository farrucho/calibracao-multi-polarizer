
# TRATAR DADOS
correct_lines = []

with open("dataRAW.txt") as file:
    for line in file:
        if len(line.split(";")) == 7 and not ';;' in line:
            correct_lines.append(line)


with open('data_corrected.csv','w') as file:
    for index,line in enumerate(correct_lines):
        try:
            if (correct_lines[index-1].split(';')[0] == correct_lines[index+1].split(';')[0]):
                if line.split(';')[0] != correct_lines[index-1].split(';')[0]:
                    correct_lines.remove(line)
                else:
                    # para remover o ultimo ;
                    if(float(line.split(';')[-2]) >= 200):
                        file.write(line[0:-2] + "\n")
        except:
            None



import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Load data
data = pd.read_csv("data_corrected.csv", delimiter=";")  

# Extract theta values and y values
theta = data.iloc[:, :-1].values  # All columns except the last one
y = data.iloc[:, -1].values  # The last column

# Define the model function
def model(params, theta):
    phi0, phi1, phi2, phi3, phi4, phi5, A = params
    theta1, theta2, theta3, theta4, theta5 = theta.T  # Unpack columns
    return A * (
        np.cos((theta1 + phi1 - phi0) * np.pi / 180) ** 2 *
        np.cos((theta2 + phi2 - theta1 - phi1) * np.pi / 180) ** 2 *
        np.cos((theta3 + phi3 - theta2 - phi2) * np.pi / 180) ** 2 *
        np.cos((theta4 + phi4 - theta3 - phi3) * np.pi / 180) ** 2 *
        np.cos((theta5 + phi5 - theta4 - phi4) * np.pi / 180) ** 2
    )

# Define the loss function (Chi²)
def loss_function(params):
    y_pred = model(params, theta)
    residuals = y - y_pred
    return np.sum(residuals ** 2)  # Minimize the sum of squared residuals

# Initial guesses
initial_guess = [25, 5, 5, 5, 5, 5, 315]

# Define bounds for parameters: phi ∈ [-180, 180], A ≥ 0
bounds = [(-180, 180)] * 6 + [(0, 600)]  # Last element is for A

# Perform optimization
result = minimize(loss_function, initial_guess, method="L-BFGS-B", bounds=bounds)

# Extract optimized parameters
phi0_opt, phi1_opt, phi2_opt, phi3_opt, phi4_opt, phi5_opt, A_opt = result.x

# Compute final Chi²
chi2 = loss_function(result.x)
ndf = len(y) - len(initial_guess)
chi2_ndf = chi2 / ndf

# Print results
print(f"Chi²: {chi2:.4f}")
print(f"ndf: {ndf}")
print(f"Chi²/ndf: {chi2_ndf:.4f}")
print(f"Optimized Parameters:\nPhi0: {phi0_opt:.4f}\nPhi1: {phi1_opt:.4f}\nPhi2: {phi2_opt:.4f}\n"
      f"Phi3: {phi3_opt:.4f}\nPhi4: {phi4_opt:.4f}\nPhi5: {phi5_opt:.4f}\nA: {A_opt:.4f}")

# Setup para intensidade máxima
print(f"Setup para varrimento com intensidade máxima:\n"
      f"{phi0_opt-phi1_opt:.4f}\n"
      f"{phi0_opt-phi2_opt:.4f}\n"
      f"{phi0_opt-phi3_opt:.4f}\n"
      f"{phi0_opt-phi4_opt:.4f}\n"
      f"{phi0_opt-phi5_opt:.4f}")
