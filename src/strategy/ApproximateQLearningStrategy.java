package strategy;

import java.util.ArrayList;
import java.util.Random;

import agent.AgentAction;
import motor.PacmanGame;
import neuralNetwork.TrainExample;


public class ApproximateQLearningStrategy extends QLearningStrategy{

	double[] weights;
	int d; // nombre de features

	
	public ApproximateQLearningStrategy(double epsilon, double gamma, double alpha, int sizeMazeX, int sizeMazeY) {
		super(epsilon, gamma, alpha, sizeMazeX, sizeMazeY);
		this.d = 13;
		this.weights = new double[d+1]; // d features + 1 pour le biais en f0 qui est toujours égal à 1
		// Initialisation des poids aléatoirement avec une valeur très faible
		for(int i = 0; i < d+1; i++) {
			weights[i] = Math.random() * 0.01;
		}
	}

	
	@Override
	public AgentAction chooseAction(PacmanGame state) {
		ArrayList<AgentAction> legalActions = state.getLegalPacmanActions();
        AgentAction bestAction = null;
        double bestValue = Double.NEGATIVE_INFINITY;

		double MAX_Q_VALUE = 1000;

        for (AgentAction action : legalActions) {
            double[] features = computeFeatures(state, action);
            double qValue = 0;
			for (int i = 0; i <= d; i++) {
				qValue += weights[i] * features[i];
				if (qValue > MAX_Q_VALUE) {
					qValue = MAX_Q_VALUE;  
					break;
				}
			}
            if (qValue > bestValue) {
                bestValue = qValue;
                bestAction = action;
            }
        }

        // Choix epsilon-greedy de l'action, pour explorer avec une probabilité epsilon
        return chooseEpsilonGreedy(bestAction, legalActions);
		
	}

	private void normalizeWeights() {
		double norm = 0.0;
		for (int i = 0; i <= d; i++) {
			norm += weights[i] * weights[i];
		}
		norm = Math.sqrt(norm);
		for (int i = 0; i <= d; i++) {
			weights[i] /= norm;
		}
	}
	


	private double[] computeFeatures(PacmanGame state, AgentAction action) {
		double[] features = new double[d+1];  // Supposons que d soit 3 pour les trois caractéristiques plus le biais
		features[0] = 1; // f0 est toujours égal à 1 (biais)
	
		// Simuler l'action pour obtenir un nouvel état hypothétique
		if (state == null || action == null) {
			System.out.println("State or action is null");
			return features;
		}


		PacmanGame hypotheticalState = state;
		int pacmanX = hypotheticalState.getPacmanX() + action.get_vx();
		int pacmanY = hypotheticalState.getPacmanY() + action.get_vy();
	
		// features[1] = bool si pacman mange une gomme après l'action
		features[1] = hypotheticalState.isGumAtPosition(pacmanX, pacmanY) ? 1 : 0;
	
		// features[2] = nb de coup pour manger une gomme après l'action (approximation simple ici)
		features[2] = estimateStepsToNearestGum(hypotheticalState, pacmanX, pacmanY);
	
		// features[3] = nb de fantomes apres l'action au voisinage de pacman
		features[3] = countGhostsNearPacman(hypotheticalState, pacmanX, pacmanY, 1); // 1 représente la distance de proximité

		// features[4] = la distance de la capsule la plus proche
		features[4] = estimateStepsToNearestCapsule(hypotheticalState, pacmanX, pacmanY);

		// features[5] = bool si pacman mange une capsule après l'action
		features[5] = hypotheticalState.isCapsuleAtPosition(pacmanX, pacmanY) ? 1 : 0;

		// features[6] = bool si pacman mange un fantome après l'action
		features[6] = hypotheticalState.isGhostAtPosition(pacmanX, pacmanY) && hypotheticalState.isGhostsScarred() ? 1 : 0;

		// features[7] = nbDeCheminsLibres après l'action
		features[7] = this.getNbFreePath(pacmanX, pacmanY, hypotheticalState);

		// features[8] = bool si pacman mange un fantome effrayé après l'action
		features[8] = hypotheticalState.isGhostsScarred() ? 1 : 0;

		// features[9] = distance de pacman au fantome le plus proche
		features[9] = distanceToNearestGhost(hypotheticalState, pacmanX, pacmanY);

		// features[10] = calcul du potentiel score après l'action
		features[10] = hypotheticalState.getScore();

		// features[11] = distance de la super gomme la plus proche
		features[11] = estimateStepsToNearestSuperGum(hypotheticalState, pacmanX, pacmanY);

		// features[12] = bool si pacman mange une super gomme après l'action
		features[12] = hypotheticalState.isGumAtPosition(pacmanX, pacmanY) ? 1 : 0;

		// features[13] = nb de food après l'action
		features[13] = hypotheticalState.getNbFood();

		// Normalisation des valeurs des caractéristiques
		double MAX_FEATURE_VALUE = 1000;
		for (int i = 0; i <= d; i++) {
			if (features[i] > MAX_FEATURE_VALUE) {
				features[i] = MAX_FEATURE_VALUE;
			}
		}

	
		return features;
	}

	private int estimateStepsToNearestSuperGum(PacmanGame state, int x, int y) {
		// Implémentez une recherche pour trouver la distance au plus proche super gomme
		int minDistance = Integer.MAX_VALUE;
		for (int i = 0; i < state.getMaze().getSizeX(); i++) {
			for (int j = 0; j < state.getMaze().getSizeY(); j++) {
				if (state.isCapsuleAtPosition(i, j)) {
					int distance = Math.abs(x - i) + Math.abs(y - j);
					if (distance < minDistance) {
						minDistance = distance;
					}
				}
			}
		}
		return minDistance;
	}
	
	private int estimateStepsToNearestGum(PacmanGame state, int x, int y) {
		// Implémentez une recherche pour trouver la distance au plus proche pacgomme
		int minDistance = Integer.MAX_VALUE;
		for (int i = 0; i < state.getMaze().getSizeX(); i++) {
			for (int j = 0; j < state.getMaze().getSizeY(); j++) {
				if (state.isGumAtPosition(i, j)) {
					int distance = Math.abs(x - i) + Math.abs(y - j);
					if (distance < minDistance) {
						minDistance = distance;
					}
				}
			}
		}
		return minDistance;
	}

	private int distanceToNearestGhost(PacmanGame state, int x, int y) {
		// Implémentez une recherche pour trouver la distance au plus proche fantome
		int minDistance = Integer.MAX_VALUE;
		for (int i = 0; i < state.getMaze().getSizeX(); i++) {
			for (int j = 0; j < state.getMaze().getSizeY(); j++) {
				if (state.isGhostAtPosition(i, j)) {
					int distance = Math.abs(x - i) + Math.abs(y - j);
					if (distance < minDistance) {
						minDistance = distance;
					}
				}
			}
		}
		return minDistance;
	}

	private int getNbFreePath(int x, int y, PacmanGame state) {
		int nbFreePath = 0;
		if(!state.isWallAtPosition(x+1, y)) {
			nbFreePath++;
		}
		if(!state.isWallAtPosition(x-1, y)) {
			nbFreePath++;
		}
		if(!state.isWallAtPosition(x, y+1)) {
			nbFreePath++;
		}
		if(!state.isWallAtPosition(x, y-1)) {
			nbFreePath++;
		}
		return nbFreePath;
	}

	private int estimateStepsToNearestCapsule(PacmanGame state, int x, int y) {
		// Implémentez une recherche pour trouver la distance au plus proche capsule
		int minDistance = Integer.MAX_VALUE;
		for (int i = 0; i < state.getMaze().getSizeX(); i++) {
			for (int j = 0; j < state.getMaze().getSizeY(); j++) {
				if (state.isCapsuleAtPosition(i, j)) {
					int distance = Math.abs(x - i) + Math.abs(y - j);
					if (distance < minDistance) {
						minDistance = distance;
					}
				}
			}
		}
		return minDistance;
	}
	
	private int countGhostsNearPacman(PacmanGame state, int x, int y, int distance) {
		int count = 0;
		for (int dx = -distance; dx <= distance; dx++) {
			for (int dy = -distance; dy <= distance; dy++) {
				if (state.isGhostAtPosition(x + dx, y + dy)) {
					count++;
				}
			}
		}
		return count;
	}
	


	public AgentAction chooseEpsilonGreedy(AgentAction bestAction, ArrayList<AgentAction> legalActions) {
		Random rand = new Random();
		double roll = rand.nextDouble(); // Génère un nombre aléatoire entre 0 et 1

		if (roll < this.current_epsilon) { // Avec une probabilité epsilon, explorer
			int randomIndex = rand.nextInt(legalActions.size());
			return legalActions.get(randomIndex);
		} else {
			return bestAction;
		}
	}


	
	@Override
	public void update(PacmanGame state, PacmanGame nextState, AgentAction action, double reward, boolean isFinalState) {
		
		double[] features = computeFeatures(state, action);
		//Calcul de la Valeur Q Actuelle :
		double qValue = 0;
		for (int i = 0; i <= d; i++) {
			qValue += weights[i] * features[i];
		}
	
		double maxQNext = 0;
		// Calcul de la Valeur Q Maximale de l'État Suivant :
		if (!isFinalState) {
			for (AgentAction nextAction : nextState.getLegalPacmanActions()) {
				double[] nextFeatures = computeFeatures(nextState, nextAction);
				double qNext = 0;
				for (int j = 0; j <= d; j++) {
					qNext += weights[j] * nextFeatures[j];
				}
				if (qNext > maxQNext) {
					maxQNext = qNext;
				}
			}
		}




		// Calcul de la cible de la mise à jour Q-learning
		double target = reward + gamma * maxQNext;
		// Calcul de l'Erreur (Delta) :
		// L'erreur (delta) est la différence entre la cible et la valeur Q actuelle estimée. Cette erreur mesure combien la prédiction actuelle était incorrecte par rapport à ce qui semble maintenant être le meilleur résultat possible compte tenu de la nouvelle information.
		double delta = target - qValue;
		

	
		// Update weights
		for (int i = 0; i <= d; i++) {
			weights[i] += this.learningRate * delta * features[i];
		}

		normalizeWeights();

	}
	
	
	
	@Override
	public void learn(ArrayList<TrainExample> trainExamples) {
		// Not used here	
	}
	
	
	
	
	
	

	
	

}
