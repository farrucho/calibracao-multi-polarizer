
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
                    file.write(line[0:-2] + "\n")
        except:
            None


# LER DADOS
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Load data from CSV
data = pd.read_csv("data_corrected.csv", delimiter=";")  # Adjust delimiter if needed

# Extract theta values and y values
theta = data.iloc[:, :-1].values  # All columns except the last one (theta1 to theta5)
y = data.iloc[:, -1].values  # The last column (y)

# Define the model function
def model(theta_flat, phi1, phi2, phi3, phi4, phi5, A):
    theta1, theta2, theta3, theta4, theta5 = theta_flat.T  # Unpack columns
    return A * (
        np.cos((theta1 - phi1)*np.pi/180) ** 2 *
        np.cos((theta2 - phi2)*np.pi/180) ** 2 *
        np.cos((theta3 - phi3)*np.pi/180) ** 2 *
        np.cos((theta4 - phi4)*np.pi/180) ** 2 *
        np.cos((theta5 - phi5)*np.pi/180) ** 2
    )

# Initial guesses for parameters (phis and A)
initial_guess = [25, 25, 25, 25, 25, 100]  # Adjust if necessary


# Fit the model
params, covariance = curve_fit(model, theta, y, p0=initial_guess)

# Extract optimized parameters
phi1_opt, phi2_opt, phi3_opt, phi4_opt, phi5_opt, A_opt = params


# Compute predicted values
y_pred = model(theta, *params)

# Assume uniform errors (sigma = 1) if not given
sigma = np.ones_like(y)  # You can replace with actual errors if known

# Compute χ²
chi2 = np.sum(((y - y_pred) / sigma) ** 2)

# Degrees of freedom (ndf)
ndf = len(y) - len(params)

# Compute χ²/ndf
chi2_ndf = chi2 / ndf

print(f"Chi²: {chi2}")
print(f"ndf: {ndf}")
print(f"Chi²/ndf: {chi2_ndf:.4f}")


# Print results
print(f"Optimized Parameters:\nPhi1: {phi1_opt}\nPhi2: {phi2_opt}\nPhi3: {phi3_opt}\nPhi4: {phi4_opt}\nPhi5: {phi5_opt}\nA: {A_opt}\nA_lux:{A_opt*75/225}")

