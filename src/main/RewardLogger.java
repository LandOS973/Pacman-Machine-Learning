package main;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

public class RewardLogger {
    private String filePath;
    private double trainReward;
    private double testReward;
    private int generation;

    public RewardLogger(int strategyID, int level) {
        switch (strategyID) {
            case 0:
                this.filePath = "logs/TabularQLearning" + level + ".csv";
                break;
            case 1:
                this.filePath = "logs/ApproximateQLearningWithLinearModelLevel" + level + ".csv";
                break;
            case 2:
                this.filePath = "logs/ApproximateQLearningWithNNLevel" + level + ".csv";
                break;
            case 3:
                this.filePath = "logs/DeepQLearningLevel" + level + ".csv";
                break;
            default:
                this.filePath = "logs/UnknownStrategyLevel" + level + ".csv";
                break;
        }
        initFile();
    }

    private void initFile() {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filePath, false))) {
            writer.println("Generation,TrainReward,TestReward");  // Header for the CSV file
        } catch (IOException e) {
            System.out.println("Error initializing the CSV file: " + e.getMessage());
        }
    }

    public void saveRewardsToFile() {
        if(generation % 50 == 0) {
            try (PrintWriter writer = new PrintWriter(new FileWriter(filePath, true))) {
                writer.printf("%d,%d,%d%n", generation, (int) trainReward, (int) testReward);
            } catch (IOException e) {
                System.out.println("Error writing to the CSV file: " + e.getMessage());
            }
        }
    }

    // Getters and Setters for trainReward and testReward
    public void setTrainReward(double trainReward) {
        this.trainReward = trainReward;
    }

    public void setTestReward(double testReward) {
        this.testReward = testReward;
    }

    public void setGeneration(int generation) {
        this.generation = generation;
    }
}
