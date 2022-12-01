import java.util.*;
class Full_Scale {
    public static void main(String args[]) {
        Scanner scan = new Scanner(System.in);
        int num[] = { 1, 5, 3, 6, 7 }, target, count = 0;

        System.out.println("Enter Target: ");
        target = scan.nextInt();

        for (int i = 0; i < num.length - 1; i++){
            for (int j = i; j < num.length - 1; j++ ){
                if(num[i] + num[j+1] == target){
                    System.out.println("Pair/s: " + num[i] + "," + num[j+1] );
                }
           }
        }

        
    }
}