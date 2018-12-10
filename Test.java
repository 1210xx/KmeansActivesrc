package classification;

import java.util.Arrays;

public class Test {

	public Test() {
		// TODO Auto-generated constructor stub
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		int[] paraBlock = {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1};
		int paraK = 2; 
		int[] tempCluster = paraBlock;
		int[] tempCollumIndex = new int[paraK];
		Arrays.fill(tempCollumIndex, 0);
		int[][] resultCluster = new int[paraK][];
		int k = 0;
//		Arrays.fill(resultCluster[0], 0);
//		Arrays.fill(resultCluster[1], 1);
//		for (int i = 0; i < paraK; i++) {
//			for (int j = 0; j < resultCluster[i].length; j++) {
//				resultCluster[i][j] = k++;
//			}
//		}

		for (int i = 0; i < paraK; i++) {
			for (int j = 0; j < paraBlock.length; j++) {
				if (tempCluster[j] == i) {
					resultCluster[i][tempCollumIndex[i]++] = j;
				}
			}
		}
		System.out.println(Arrays.deepToString(resultCluster));
	}

}
