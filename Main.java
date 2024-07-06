import java.awt.image.BufferedImage;
import java.io.*;
import javax.imageio.ImageIO;

public class Main{
    public static void main(String[] args) throws IOException{
        int[] sizes = {16, 16, 10};
        Network nn = new Network(3, sizes, 784);
        File[] trainFiles = new File("Training").listFiles();
        double[][] training = new double[trainFiles.length][784];
        int[] target = new int[trainFiles.length];
        System.out.println("Formatting "+trainFiles.length+" training files for backpropagation...");
        for (int i=0; i<trainFiles.length; i++){
            if (trainFiles[i].getName().charAt(0)<48||trainFiles[i].getName().charAt(0)>57){
                continue;
            }
            target[i] = trainFiles[i].getName().charAt(0)-48;
            BufferedImage image = ImageIO.read(trainFiles[i]);
            while (image==null){

            }
            for (int y=0; y<28; y++){
                for (int x=0; x<28; x++){
                    int rgb = image.getRGB(x, y);
                    int alpha = (rgb >> 24) & 0xFF;
                    int red =   (rgb >> 16) & 0xFF;
                    int green = (rgb >>  8) & 0xFF;
                    int blue =  (rgb      ) & 0xFF;
                    training[i][y*28+x] = (red+blue+green)/768.0;
                }
            }
        }
        System.out.println("Running backpropagation...");
        for (int i=0; i<2000; i++){
            nn.learn(training, target);
        }
        File[] testFiles = new File("Testing").listFiles();
        System.out.println("Testing neural network on "+testFiles.length+" files...");
        int correct = 0;
        int total = 0;
        for (int i=0; i<testFiles.length; i++){
            if (testFiles[i].getName().charAt(0)<48||testFiles[i].getName().charAt(0)>57){
                continue;
            }
            total++;
            BufferedImage image = ImageIO.read(testFiles[i]);
            while (image==null){

            }
            double[] input = new double[784];
            for (int y=0; y<28; y++){
                for (int x=0; x<28; x++){
                    int rgb = image.getRGB(x, y);
                    int alpha = (rgb >> 24) & 0xFF;
                    int red =   (rgb >> 16) & 0xFF;
                    int green = (rgb >>  8) & 0xFF;
                    int blue =  (rgb      ) & 0xFF;
                    input[y*28+x] = (red+blue+green)/768.0;
                }
            }
            double[] result = nn.predict(input);
            int max = 0;
            for (int j=0; j<10; j++){
                if (result[j]>result[max]){
                    max = j;
                }
            }
            System.out.println(testFiles[i].getName()+": "+max);
            if (testFiles[i].getName().charAt(0)-48==max){
                correct++;
            }
        }
        System.out.println("Accuracy: "+correct*100.0/total+"%");
    }
}