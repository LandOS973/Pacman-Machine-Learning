import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

def plot_rewards(filename):
    # Chargement des données à partir du fichier CSV
    data = pd.read_csv(filename, names=['Generation', 'TrainReward', 'TestReward'], header=0)
     # convert to int but not the first column
    data['Generation'] = data['Generation'].astype(int)
    data['TrainReward'] = data['TrainReward'].astype(float)
    data['TestReward'] = data['TestReward'].astype(float)
    
    # Création de la figure
    plt.figure(figsize=(10, 5))
    plt.plot(data['Generation'], data['TrainReward'], label='Train Reward')
    plt.plot(data['Generation'], data['TestReward'], label='Test Reward')
    
    # Ajout de la légende
    plt.legend()
    
    # Ajout des labels
    plt.xlabel('Generation')
    plt.ylabel('Reward')

    # Ajout du titre et des légendes
    plt.title('Comparison of Train and Test Rewards Over Generations')

    # Construction du nom de fichier pour la sortie
    output_filename = os.path.splitext(filename)[0] + '.png'
    
    # Enregistrement de la figure
    plt.savefig(output_filename)
    print(f"Plot saved as {output_filename}")

if __name__ == '__main__':
    # Vérifie si un argument de fichier est passé
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        plot_rewards(filename)
    else:
        print("Please provide the filename as an argument: ./script.py filename.csv")
