package strategy;

import java.util.ArrayList;
import java.util.Random;
import java.util.Arrays;
import java.util.Map; 

import agent.Agent;
import agent.AgentAction;
import agent.PositionAgent;
import motor.Maze;
import motor.PacmanGame;
import neuralNetwork.TrainExample;

import java.util.HashMap;


public class TabularQLearning  extends QLearningStrategy{

	// chaine de caractere qui represente un etat unique (position pacman, position des fantomes, position des points, etc)
	// "10032224" => pacgom =1 rien = 0 mur = 3 fantome = 2 pacman = 4 => plus il faut stocker l'info si les fantomes sont appeurés
	// il faudra faire une fonction pour encoder l'état et regarder si il est deja dans le dictonaire
	// avec un etat est associé un vector de double qui represente les valeurs Q pour chaque action possible
	// la table permet d'associé un etat avec une postion
	// stocker aussi le nombre de tour ou le pacman est invincible (quand il a mangé un super pacgom)
	HashMap<String, double[]> QTable;


	// S0 => etat initial [0,0,1,0] N S E O
	// S1 => etat suivant [0,0,-10,0]

	// Q(s,a) = (1-alpha) * Q(s,a) + alpha * (r + gamma * max(Q(s',a'))) s' et a' le prochain etat et action

	int sizeMazeX;
	int sizeMazeY;




	public TabularQLearning( double epsilon, double gamma, double alpha,  int sizeMazeX, int sizeMazeY, int nbWalls) {
		
		super( epsilon, gamma, alpha, sizeMazeX, sizeMazeY);

		this.sizeMazeX = sizeMazeX;
		this.sizeMazeY = sizeMazeY;

		System.out.println("sizeX labyrinth " + this.sizeMazeX);
		System.out.println("sizeY labyrinth " + this.sizeMazeY);
		
		int numberCellsWithoutWall = sizeMazeX*sizeMazeY - nbWalls;
				
		System.out.println("NumberCells without wall " + numberCellsWithoutWall);

		int numberStates =  (int) Math.pow( 4, numberCellsWithoutWall);

		System.out.println("Max number different states " + numberStates);

		QTable = new HashMap<>();

	}

	// Exercice 1 : implémentation de la stratégie Tabular Q-learning
	// L’objectif de ce premier exercice est d’implémenter la stratégie Tabular Q-learning vue dans le
	// cours 2. Tout est à faire dans la classe TabularQLearning qui étend la classe QLearningStrategy et
	// implémente l’interface Strategy.
	// 1. Dans le constructeur TabularQLearning, créer une table Q sous la forme d’un dictionnaire
	// HashMap<String, double[]>. L’idée sera de représenter un état du jeu par une chaîne de
	// caractères qui sera la clé de ce dictionnaire. Pour chaque clé, on associera une vecteur de
	// double dont la taille correspond au nombre d’actions possibles du Pacman (ici 4 actions
	// possibles : nord, sud, est et ouest).

	// 0 => rien
	// 1 => pacgomme
	// 2 => pacman
	// 3 => capsule
	// 4 => ghost
	// 5 => mur

	public String encodeState(PacmanGame state){
		String s = "";

		for(int i = 0; i < this.sizeMazeX; i++){
			for(int j = 0; j < this.sizeMazeY; j++){
				if(state.isGumAtPosition(i, j)){
					s += "1";
				}else if(state.isPacmanAtPosition(i, j)){
					s += "2";
				}else if(state.isCapsuleAtPosition(i, j)){
					s += "3";
				}else if(state.isGhostAtPosition(i, j)){
					s += "4";
				}else if(state.isWallAtPosition(i, j)){
					s += "5";
				}else{
					s += "0";
				}
			}
		}
		return s;

	}

	public void encodeQtable(PacmanGame state) {
		String s = encodeState(state);
		if(!QTable.containsKey(s)){
			double[] actions = new double[4];
			for(int i = 0; i < 4; i++){
				actions[i] = 0;
			}
			QTable.put(s, actions);
		}
	}





	@Override
	public AgentAction chooseAction(PacmanGame state) {
		encodeQtable(state);
		// system ouf la q table
		for (Map.Entry<String, double[]> entry : QTable.entrySet()) {
			String key = entry.getKey();
			double[] value = entry.getValue();
			System.out.println(key + " " + Arrays.toString(value));
		}
		System.out.println();
		if(Math.random() < this.current_epsilon){
			// choix random
			Random r = new Random();
			int action = r.nextInt(4);
			AgentAction A = new AgentAction(action);
			while (!state.isLegalMovePacman(A)) {
				int action2 = r.nextInt(4);
				A = new AgentAction(action2);
			}
			return A;
		}else{
			// prendre la meilleure action
			String s = encodeState(state);
			double[] actions = QTable.get(s);
			int index = 0;
			double max = actions[0];
			for(int i = 1; i < 4; i++){
				if(actions[i] > max && state.isLegalMovePacman(new AgentAction(i))){
					max = actions[i];
					index = i;
				}
			}
			AgentAction A = new AgentAction(index);
			return A;
		}
	}





	@Override
	public void update(PacmanGame state, PacmanGame nextState, AgentAction action, double reward, boolean isFinalState) {
		String currentStateKey = encodeState(state);
		String nextStateKey = encodeState(nextState);
		
		if (!QTable.containsKey(currentStateKey)) {
			QTable.put(currentStateKey, new double[]{0, 0, 0, 0}); 
		}
		if (!QTable.containsKey(nextStateKey)) {
			QTable.put(nextStateKey, new double[]{0, 0, 0, 0});
		}
		
		double[] currentQValues = QTable.get(currentStateKey);
		double[] nextQValues = QTable.get(nextStateKey);
		
		double maxNextQ = isFinalState ? 0 : Arrays.stream(nextQValues).max().getAsDouble(); 
		int actionIndex = action.get_idAction();
		
		// Mise à jour Q-value en utilisant la formule Q-learning
		currentQValues[actionIndex] = (1 - learningRate) * currentQValues[actionIndex] + learningRate * (reward + gamma * maxNextQ);
	}



	@Override
	public void learn(ArrayList<TrainExample> trainExamples) {
		// TODO Auto-generated method stub
	}







}
