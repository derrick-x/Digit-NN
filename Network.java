

public class Network{
    int layers;
    int[] sizes;
    int inSize;
    double[][][] weights;
    public Network(int l, int[] s, int i){
        layers = l; sizes = s; inSize = i;
        weights = new double[layers][][];
        weights[0] = new double[sizes[0]][inSize];
        for (int j=1; j<layers; j++){
            weights[j] = new double[sizes[j]][sizes[j-1]];
        }
        for (int j=0; j<weights.length; j++){
            for (int k=0; k<weights[j].length; k++){
                for (int m=0; m<weights[j][k].length; m++){
                    weights[j][k][m] = Math.random()*2-1;
                }
            }
        }
    }
    public double[] predict(double[] input){
        return run(input)[layers-1];
    }
    private double[][] run(double[] input){
        double[][] network = new double[layers][];
        network[0] = new double[sizes[0]];
            for (int j=0; j<sizes[0]; j++){
                double sum = 0;
                for (int k=0; k<inSize; k++){
                    sum+=input[k]*weights[0][j][k];
                }
                network[0][j] = sigmoid(sum);
            }
        for (int i=1; i<layers; i++){
            network[i] = new double[sizes[i]];
            for (int j=0; j<sizes[i]; j++){
                double sum = 0;
                for (int k=0; k<sizes[i-1]; k++){
                    sum+=network[i-1][k]*weights[i][j][k];
                }
                network[i][j] = sigmoid(sum);
            }
        }
        return network;
    }
    private double[][] runNoSquish(double[] input){
        double[][] network = new double[layers][];
        network[0] = new double[sizes[0]];
            for (int j=0; j<sizes[0]; j++){
                double sum = 0;
                for (int k=0; k<inSize; k++){
                    sum+=input[k]*weights[0][j][k];
                }
            }
        for (int i=1; i<layers; i++){
            network[i] = new double[sizes[i]];
            for (int j=0; j<sizes[i]; j++){
                double sum = 0;
                for (int k=0; k<sizes[i-1]; k++){
                    sum+=sigmoid(network[i-1][k])*weights[i][j][k];
                }
            }
        }
        return network;
    }
    public void learn(double[][] training, int[] target){
        double[][][] change = new double[weights.length][][];
        for (int i=0; i<weights.length; i++){
            change[i] = new double[weights[i].length][];
            for (int j=0; j<weights[i].length; j++){
                change[i][j] = new double[weights[i][j].length];
            }
        }
        for (int i=0; i<training.length; i++){
            double[][][] backprop = learn(training[i], target[i]);
            for (int j=0; j<weights.length; j++){
                for (int k=0; k<weights[j].length; k++){
                    for (int l=0; l<weights[j][k].length; l++){
                        change[j][k][l]+=backprop[j][k][l];
                    }
                }
            }
        }
        for (int i=0; i<change.length; i++){
            for (int j=0; j<change[i].length; j++){
                for (int k=0; k<change[i][j].length; k++){
                    weights[i][j][k]+=change[i][j][k]*0.01;
                }
            }
        }
    }
    private double[][][] learn(double[] training, int target){
        double[][] network = run(training);
        double[][] noSquish = runNoSquish(training);
        double[][][] change = new double[weights.length][][];
        for (int i=0; i<weights.length; i++){
            change[i] = new double[weights[i].length][];
            for (int j=0; j<weights[i].length; j++){
                change[i][j] = new double[weights[i][j].length];
            }
        }
        double[] direction = new double[sizes[layers-1]];
        for (int i=0; i<sizes[layers-1]; i++){
            direction[i]-=network[layers-1][i]-(i==target?1:0);
        }
        for (int i=layers-1; i>0; i--){
            double[] nextDirection = new double[sizes[i-1]];
            for (int j=0; j<sizes[i]; j++){
                for (int k=0; k<sizes[i-1]; k++){
                    change[i][j][k]+=direction[j]*sigDeriv(noSquish[i][j])*network[i-1][k];
                    nextDirection[k]+=direction[j]*sigDeriv(noSquish[i][j])*weights[i][j][k];
                }
            }
            direction = nextDirection;
        }
        for (int j=0; j<sizes[0]; j++){
            for (int k=0; k<inSize; k++){
                change[0][j][k]+=direction[j]*sigDeriv(noSquish[0][j])*training[k];
            }
        }
        return change;
    }
    private double sigmoid(double num){
        return 1.0/(1+Math.pow(Math.E, num*-1));
    }
    private double sigDeriv(double num){
        return sigmoid(num)*(1-sigmoid(num));
    }
}