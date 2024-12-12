import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;

public class App_csv {
    static short[] input;
    static short[] actual;

    static {
        try {
            input = readShortArrayFromCSV("classified.csv");
            actual = readShortArrayFromCSV("actual.csv");
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static short[] readShortArrayFromCSV(String filename) throws Exception {
        BufferedReader br = new BufferedReader(new FileReader(filename));
        String line = br.readLine();
        br.close();
        String[] tokens = line.split(",");
        short[] array = new short[tokens.length];
        for(int i = 0; i < tokens.length; i++) {
            array[i] = Short.parseShort(tokens[i].trim());
        }
        return array;
    }

    public static void main(String[] args) throws Exception {
        String[] p = {"Without Peter", "With Peter", "Without Peter", "With Peter"};
        PrintWriter writer = new PrintWriter("what.txt", "UTF-8");
        writer.println("#!/bin/sh");
        writer.println("ffmpeg -i ~/Downloads/S2E18.mkv -vf \"");
        
        short state = input[0];
        if(state != actual[0])
            state += 2;
        int start = 0;
        
        for(int i = 0; i < input.length; i++){
            short in = input[i];
            if(in != actual[i])
                in += 2;
            if(in != state){
                if(state == 0 || state == 1)
                    writer.printf("drawtext=text=%s:enable='between(n,%d,%d)':x=10:y=10:fontsize=48:fontcolor=green", p[state], start*10, i*10-1);
                if(state == 2 || state == 3)
                    writer.printf("drawtext=text=%s:enable='between(n,%d,%d)':x=10:y=10:fontsize=48:fontcolor=red", p[state], start*10, i*10-1);
                if(i != input.length-1)
                    writer.print(',');
                writer.println();
                start = i;
                state = in;
            }
            if(i == input.length-1){
                if(state == 0 || state == 1)
                    writer.printf("drawtext=text=%s:enable='between(n,%d,%d)':x=10:y=10:fontsize=48:fontcolor=green", p[state], start*10, i*10+9);
                if(state == 2 || state == 3)
                    writer.printf("drawtext=text=%s:enable='between(n,%d,%d)':x=10:y=10:fontsize=48:fontcolor=red", p[state], start*10, i*10+9);
                writer.println();
            }
        }
        
        writer.println("\" -codec:a copy output4.mkv");
        writer.close();
    }
}
