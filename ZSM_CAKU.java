package classification;

import java.io.FileReader;
import java.math.BigDecimal;
import java.util.*;
import weka.*;
import weka.core.Instances;

public class ZSM_CAKU {

	static Random random = new Random();

	/**
	 * ���ݶ�ȡ
	 */
	Instances data;

	/**
	 * 1 represents bought and 2 represents predicted.
	 */
	int[] instanceStates;

	/**
	 * data's labels
	 */
	int[] labels;

	/**
	 * all centers attributes
	 */
	double[][] currentCenters;

	/**
	 * tcost:teach cost
	 */
	double tCost;

	/**
	 * mcost:mis-classified Cost
	 */
	double[] mCost;

	/**
	 *******************************
	 * main function
	 *******************************
	 */
	public static void main(String args[]) {

		double[] mCost = { 2, 4 };

		double avgCost = 0;
		ZSM_CAKU tempLeaner = new ZSM_CAKU("data/CAKU.arff", mCost, 1);
		tempLeaner.learningTest();
		avgCost += tempLeaner.totalCost();
		System.out.println("OK");
		System.out.println("onceOfCost" + avgCost);
	}// Of main

	/**
	 **************************************
	 * compute totalCost of all data
	 * 
	 * @param cost:total cost 
	 * @return sum cost of mis-classification and teach
	 **************************************
	 */
	public double totalCost() {
		double cost = 0;
		double teachCostNumber = 0;
		double misClassCost = 0;

		for (int i = 0; i < data.numInstances(); i++) {
			if (instanceStates[i] == 1) {
				cost += tCost;
				teachCostNumber++;
			} else {
				if (labels[i] == 0 && (int) data.instance(i).classValue() == 1) {
					cost += mCost[0];// ������� labels�����ʾ����ÿ��Instance���ֳ��ĸ���
					misClassCost += mCost[0];
				} else if (labels[i] == 1 && (int) data.instance(i).classValue() == 0) {
					cost += mCost[1];
					misClassCost += mCost[1];
				} // Of if
			} // Of if else
		} // Of for i

		System.out.println("teachCost(numbers):" + teachCostNumber);
		System.out.println("misclassifiedCost:" + misClassCost);
		return cost;
	}// Of totalCost

	/**
	 **************************************
	 * Constructor
	 * 
	 * @param paraFilename:fileName
	 * @param misClassifiedCost[]:misClassifiedCost
	 * @param teachCost:teach cost
	 **************************************
	 */
	public ZSM_CAKU(String paraFilename, double misClassifiedCost[], int teachCost) {
		mCost = misClassifiedCost;
		tCost = teachCost;
		data = null;
		try {
			FileReader fileReader = new FileReader(paraFilename);
			data = new Instances(fileReader);
			fileReader.close();
			data.setClassIndex(data.numAttributes() - 1);
			// System.out.println(data);
		} catch (Exception ee) {
			// System.out.println("Cannot read the file: " + paraFilename + "\r\n" + ee);
			System.exit(0);
		} // Of try

		// Initialize
		instanceStates = new int[data.numInstances()];
		labels = new int[data.numInstances()];
		Arrays.fill(labels, -1);
	}// Of the constructor

	/**
	 **************************************
	 * Mark all data with index and learning. 
	 **************************************
	 */
	void learningTest() {
		int[] originalBlock = new int[data.numInstances()];// ��ʼ
		for (int i = 0; i < originalBlock.length; i++) {
			originalBlock[i] = i;
		} // Of for i

		learning(originalBlock);// �����ֻ��ԭ���ݵ��±�

		System.out.println("instanceStates: " + Arrays.toString(instanceStates));
		System.out.println("labels: " + Arrays.toString(labels));

	}// Of learningTest

	/**
	 **************************************
	 * learn data
	 * 
	 * @param paraBlock The block of instance
	 **************************************
	 */
	void learning(int[] paraBlock) {
		
		System.out.println("Now learn these data:" + Arrays.toString(paraBlock));
		
		int blockSize = paraBlock.length;
		int firstLabel = -1;
		int IndexOfFirstLabel = -1;
		int secondLabel = -1;
		int IndexOfSecondLabel = -1;
		
		// ����С��5ֱ�ӹ���
		if (blockSize <= 5) {
			for (int i = 0; i < blockSize; i++) {
				instanceStates[paraBlock[i]] = 1;
				labels[paraBlock[i]] = (int) data.instance(paraBlock[i]).value(data.numAttributes() - 1);
			} // Of for i
			return;
		} // Of if

		// �ҵ���һ���Ѿ�����ĵ�
		for (int i = 0; i < blockSize; i++) {
			if (instanceStates[paraBlock[i]] == 1) {
				IndexOfFirstLabel = i;
				firstLabel = (int) data.instance(paraBlock[i]).value(data.numAttributes() - 1);
				break;
			} // Of if
		} // Of for i

		//����ǩ��ֱͬ�ӽ��з���
		if (IndexOfFirstLabel != -1) {
			for (int i = IndexOfFirstLabel + 1; i < blockSize; i++) {
				if (instanceStates[paraBlock[i]] == 1) {
					IndexOfSecondLabel = i;
					secondLabel = (int) data.instance(paraBlock[i]).value(data.numAttributes() - 1);
					if (secondLabel != firstLabel) {
						System.out.println("Now split.");
						splitAndLearn(paraBlock, IndexOfSecondLabel, IndexOfFirstLabel);
						return;
					} // Of if
				} // Of if(instanceState) 
			} // Of for i
		} // Of if(IndexOfFirstLabel)

		/**
		 **************************************
		 * find the number of need to buy
		 * 
		 * @param needToBuy the number of need to buy instance
		 **************************************
		 */
		int needToBuyNum = findRepresentatives(paraBlock, firstLabel);
		List<Integer> pointIndex = new ArrayList<Integer>();

		if (IndexOfFirstLabel != -1) {
			pointIndex.add(IndexOfFirstLabel);
			for(int i=IndexOfFirstLabel+1;i<blockSize;i++) {
				if(labels[paraBlock[i]]==1) {
					pointIndex.add(i);
				}
			}
		}
		else {
			pointIndex.add(0); // ����ǰ���û���Ѿ�����㣬���ڿ�ʼ˳�����µ�һ�������
			instanceStates[paraBlock[pointIndex.get(0)]] = 1;
			labels[paraBlock[pointIndex.get(0)]] = (int) data.instance(paraBlock[pointIndex.get(0)])
					.value(data.numAttributes() - 1);
		} // Of if else

		/**
		 *******************
		 * Now find other farthest index in this block.
		 ******************* 
		 */
		while (pointIndex.size() <= (needToBuyNum - 1)) {

			// ���ص���paraBlock�ڲ��±꣬��������ԭʼ�ı��
			int zuiyuandian = computeFathestPoint(paraBlock, pointIndex);

			// ������Զ��ı�ǩ
			instanceStates[paraBlock[zuiyuandian]] = 1;
			labels[paraBlock[zuiyuandian]] = (int) data.instance(paraBlock[zuiyuandian])
					.value(data.numAttributes() - 1);

			if (labels[paraBlock[zuiyuandian]] != labels[paraBlock[pointIndex.get(0)]]) {
				splitAndLearn(paraBlock, pointIndex.get(0), zuiyuandian);
				return;
			} else {
				pointIndex.add(zuiyuandian);
			} // Of if and else
		} // of while

		//Predict others in this block.
		for (int i = 0; i < paraBlock.length; i++) {
			if (instanceStates[paraBlock[i]] != 1) {
				instanceStates[paraBlock[i]] = 2;
				labels[paraBlock[i]] = labels[paraBlock[pointIndex.get(0)]];
			} // Of if
		} // Of for i
		return;
	}// of learning

	/**
	 **************************************
	 * @param paraBlock
	 * @param           firstlabelIndex:paraBlock's First index of instance which
	 *                  has label
	 * @param           differentLabelIndex:The index of instance which has
	 *                  different label with first instance
	 **************************************
	 */
	public void splitAndLearn(int[] paraBlock, int firstlabelIndex, int differentLabelIndex) {
		// Step 1. Split
		int[] tempClutering = cluster(firstlabelIndex, differentLabelIndex, paraBlock);
		int tempFirstBlockSize = 0;
		for (int i = 0; i < tempClutering.length; i++) {
			if (tempClutering[i] == 0) {
				tempFirstBlockSize++;
			} // Of if
		} // Of for i

		int[][] tempBlocks = new int[2][];
		tempBlocks[0] = new int[tempFirstBlockSize];
		tempBlocks[1] = new int[paraBlock.length - tempFirstBlockSize];

		int[] tempCounters = { 0, 0 };

		for (int i = 0; i < tempClutering.length; i++) {
			tempBlocks[tempClutering[i]][tempCounters[tempClutering[i]]++] = paraBlock[i];
		} // Of for i

		System.out.println("SplittedAndLearn into two blocks: " + Arrays.toString(tempBlocks[0]) + "\r\n"
				+ Arrays.toString(tempBlocks[1]));

		// Step 2. Learn
		learning(tempBlocks[0]);
		learning(tempBlocks[1]);
	}// Of splitAndLearn

	/**
	 **************************************
	 * kMeans ��ʼ������������Զ�ı�ǩ��ͬ�ĵ�
	 * 
	 * @param firstlabelIndex
	 * @param differentLabelIndex
	 * @param paraBlock
	 * @return ���飺�±������������һ���ࣿ
	 **************************************
	 */
	public int[] cluster(int firstlabelIndex, int differentLabelIndex, int[] paraBlock) {
		// Step 1. Initialize
		int tempBlockSize = paraBlock.length;
		int[] tempCluster = new int[tempBlockSize];
		double[][] tempCenters = new double[2][data.numAttributes() - 1];
		double[][] tempNewCenters = new double[2][data.numAttributes() - 1];

		// Step 2.Initialize the center of attribute
		for (int j = 0; j < data.numAttributes() - 1; j++) {
			tempNewCenters[0][j] = data.instance(paraBlock[firstlabelIndex]).value(j);
			tempNewCenters[1][j] = data.instance(paraBlock[differentLabelIndex]).value(j);
		}

		// Step 3. Cluster and compute new centers.
		while (!doubleMatricesEqual(tempCenters, tempNewCenters)) {
			// while (!Arrays.deepEquals(tempCenters, tempNewCenters)) {
			tempCenters = tempNewCenters;
			// Cluster
			for (int i = 0; i < tempBlockSize; i++) {
				double tempDistance = Double.MAX_VALUE;
				for (int j = 0; j < 2; j++) {
					double tempCurrentDistance = distance(paraBlock[i], tempCenters[j]);
					if (tempCurrentDistance < tempDistance) {
						tempCluster[i] = j;// ����ÿ���������ĸ�����
						tempDistance = tempCurrentDistance;
					} // Of cluster
				} // Of for j
			} // Of for i

			// Compute new centers
			int[] tempCounters = new int[2];
			for (int i = 0; i < tempCounters.length; i++) {
				tempCounters[i] = 0;
			} // Of for i

			tempNewCenters = new double[2][data.numAttributes() - 1];
			for (int i = 0; i < tempBlockSize; i++) {
				tempCounters[tempCluster[i]]++;// ����ÿ�������м�����
				for (int j = 0; j < data.numAttributes() - 1; j++) {
					tempNewCenters[tempCluster[i]][j] += data.instance(paraBlock[i]).value(j);
				} // Of for j
			} // Of for i

			// Average
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < data.numAttributes() - 1; j++) {
					tempNewCenters[i][j] /= tempCounters[i];
				} // Of for j
			} // Of for i
			currentCenters = tempNewCenters;// ��������������㵽���ĵ��ƽ������

		} // Of while
		for (int i = 0; i <2; i++)
			System.out.println("���ξ�������" + Arrays.toString(currentCenters[i]));

		System.out.println("ԭ����cluster�����ĸ�����" + Arrays.toString(tempCluster));
		return tempCluster;
	}// Of cluster

	/**
	 ************************************** 
	 * compute distance between two data of index
	 * 
	 * @param point��the index of all instance
	 * @param paraBlockPoint:the index of all instance
	 * @return Distance between two point
	 **************************************
	 */
	double computeDistance(int point, int paraBlockPoint) {
		double tempDistance = 0;
		for (int i = 0; i < data.numAttributes() - 1; i++) {
			tempDistance += Math.pow(data.instance(point).value(i) - data.instance(paraBlockPoint).value(i),2);
		} // Of for i
		return tempDistance;
	}// Of computeDistance

	/**
	 **************************************
	 * ���㵱ǰ���ݿ���Զ�����ݵ㣨�����������ĵ���Զ�ĵ㣩
	 * 
	 * @param              paraBlock:current data set
	 * @param centerPoints : the index of farthest centers in now paraBlock
	 * @return :the index of farthest center in now paraBlock
	 **************************************
	 */
	public int computeFathestPoint(int[] paraBlock, List<Integer> centerPoints) {
		int blockSize = paraBlock.length;
		int listSize = centerPoints.size();
		int tempIndexOfMaxPoint = 0;
		boolean[] isFinded = new boolean[blockSize];
		double tempMaxDistance = 0;

		// Step 1.Initialize mark of instance
		for (int i = 0; i < listSize; i++) {
			isFinded[centerPoints.get(i)] = true;
		} // Of for i

		// Step 2.find the farthest point
		for (int i = 0; i < blockSize; i++) {
			double tempDistance = 0;
			for (int j = 0; j < listSize; j++) {
				tempDistance += computeDistance(paraBlock[centerPoints.get(j)], paraBlock[i]);
			} // Of for j
			if ((tempDistance > tempMaxDistance) && (!isFinded[i])) {
				tempIndexOfMaxPoint = i;
				tempMaxDistance = tempDistance;
			} // Of for i
		} // Of for j

		System.out.println("��Զ�����������֪����Զ�����������е�������" + paraBlock[tempIndexOfMaxPoint]);
		return tempIndexOfMaxPoint;
	}// Of computeFathestPoint

	/**
	 **************************************
	 * With how many instances with the same label can we say it is pure?
	 * ����һ����������ͬһ����ǽ����������Ϊ���ࡣ
	 **************************************
	 */
	public int pureThreshold(int paraSize) {
		return (int) Math.sqrt(paraSize);
	}// Of pureThreshold

	/**
	 * 
	 * @param paraMatrix1:centers attribute;
	 * @param paraMatrix2
	 * @return true means equal
	 */
	public static boolean doubleMatricesEqual(double[][] paraMatrix1, double[][] paraMatrix2) {
		for (int i = 0; i < paraMatrix1.length; i++) {
			for (int j = 0; j < paraMatrix1[0].length; j++) {
				if (Math.abs(paraMatrix1[i][j] - paraMatrix2[i][j]) > 1e-6) {
					return false;
				} // Of if
			} // Of for j
		} // Of for i
		return true;
	}// Of doubleMatricesEqual

	/**
	 **************************************
	 * The probability of data block is pure
	 * 
	 * @param R:red ball number
	 * @param B:blue ball number
	 * @param N:sampling times
	 * @return the probability of paraBlock which is pure
	 **************************************
	 */
	public static double expectPosNum(int R, int B, int N) {
		BigDecimal fenzi = new BigDecimal("0");
		BigDecimal fenmu = new BigDecimal("0");
		for (int i = R; i <= N - B; i++) {
			BigDecimal a = A(R, i).multiply(A(B, N - i));
			fenzi = fenzi.add(a.multiply(new BigDecimal("" + i)));
			fenmu = fenmu.add(a);
			// System.out.println("fenzi:" + fenzi + ", fenmu: " + fenmu);
		} // Of for i
		return fenzi.divide(fenmu, 4, BigDecimal.ROUND_HALF_EVEN).doubleValue();
	}// Of expectPosNum

	/**
	 **************************************
	 * ���������ѧ�е�A�����±�
	 * 
	 * @param m���ϱ�
	 * @param n���±�
	 * @return resultOfA:result of A
	 **************************************
	 */
	public static BigDecimal A(int m, int n) {
		if (m > n) {
			return new BigDecimal("0");
		} // Of if
		BigDecimal resultOfA = new BigDecimal("1");
		for (int i = n - m + 1; i <= n; i++) {
			resultOfA = resultOfA.multiply(new BigDecimal(i));
		} // Of if
		return resultOfA;
	}// Of A

	/**
	 **************************************
	 * the distance between data of index and all centers
	 * 
	 * @param paraIndex:Index of all data
	 * @param paraArray:the attribute of centers
	 * @return distance
	 **************************************
	 */
	public double distance(int paraIndex, double[] paraArray) {
		double resultDistance = 0;
		for (int i = 0; i < paraArray.length; i++) {
			resultDistance += Math.pow(data.instance(paraIndex).value(i) - paraArray[i],2);
		} // Of for i
		return resultDistance;
	}// Of distance

	/**
	 **************************************
	 * Find some number of representatives given the instances.
	 **************************************
	 */
	public int findRepresentatives(int[] paraBlock, int tempFirstLabel) {
		// Step 1. How many labels do we already bought?
		int tempLabels = 0;
		for (int i = 0; i < paraBlock.length; i++) {
			if (instanceStates[paraBlock[i]] == 1) {
				tempLabels++;
			} // Of if
		} // Of for i

		// Step 2. How many labels to buy?
		int[] tempNumBuys = lookup(paraBlock.length);
		// ���ص������ǵڣ��±꣩����ʱ�飬���ٸ��ܴ�����С��
		// tempNumBuys[0]��tempNumBuys[1]�����һ����ߵڶ���� i�� ����������������ٸ���Ϊ�Ǵ��Ĳ��Ҵ�����С)
		int tempBuyLabels = 0;
		if (tempFirstLabel == 0 || tempFirstLabel == 1) {// ��tempFirstLabel�ǵ�һ����ߵڶ���
			tempBuyLabels = tempNumBuys[tempFirstLabel] - tempLabels;
			// ��������������ĸ�����ȥ�Ѿ���ĸ���
		} else {
			tempBuyLabels = Math.max(tempNumBuys[0], tempNumBuys[1]) - tempLabels;
			// �������������lookup���ص����ֵ��ȥ�Ѿ���ĸ�����
		} // Of if

		// int tempBuyLabels = pureThreshold(paraBlock.length) - tempLabels;
		if (pureThreshold(paraBlock.length) - tempBuyLabels > 0) {// ���㴿���ż�-Ҫ��ı�ǩ��Ŀ
			System.out.println("+");
		} else if (pureThreshold(paraBlock.length) - tempBuyLabels < 0) {
			System.out.println("-");
		} // Of if else
		
		System.out.println("number to buy:"+tempBuyLabels);
		return tempBuyLabels;
	}// Of findRepresentatives

	/**
	 **************************************
	 * Find the smallest cost of take numbers
	 * 
	 * @param pSize:the size of paraBlock
	 * @return the minimal cost of one's sample
	 **************************************
	 */
	private int[] lookup(int pSize) {
		// Linear Search�������ݿ��С
		double[] tmpMinCost = new double[] { 0.5 * pSize * mCost[0], 0.5 * pSize * mCost[1] };
		int[] star = new int[2];
		int[] starLoose = new int[2];
		boolean[] isFind = new boolean[2];
		double[] ra = new double[pSize + 1];
		// ra[]������ǳ鵽��i��ʱ���ж�������Ϊ�������Ǵ���
		ra[0] = 0.5;// ��һ������Ǻ���������ĸ��ʶ���0.5�����Ұ��������е��뷨��������Ķ���һ�µ���ɫ
		double[] tmpCost = new double[2];// 1���Ϊ2��Ĵ����2���Ϊ1��Ĵ�����󶼼���tcost��
		for (int i = 1; i <= pSize; i++) {
			if (pSize >= 1000) {
				ra[i] = (i + 1.0) / (i + 2.0);// ���� �����ȴ���1000�������鵽��i����ͬ��ɫʱ�򴿵ĸ�����i+1/i+2
			} else {
				ra[i] = expectPosNum(i, 0, pSize) / pSize;// ��С��1000�����ֱ�Ӽ��������ı��� ����101������ ����0-100����һ�����
															// ��ÿ������������˸��ʣ���ͣ�����80����80-100�ĺͣ����������101 ���������ĸ���
			} // ����ÿ�������ĸ����¿��ܵĴ��ۡ�
			for (int j = 0; j < 2; j++) {// 1���Ϊ2��Ĵ����2���Ϊ1��Ĵ�����󶼼���tcost��
				tmpCost[j] = (1 - ra[i]) * mCost[j] * pSize + tCost * i;// 0����1�������Ĵ���
				if (tmpCost[j] < tmpMinCost[j]) {// ����0����1��ǰ�����µ���С���ۣ����Ҽ�¼i�ĸ���
					tmpMinCost[j] = tmpCost[j];
					star[j] = i;// ���һ�����С���������i����������star[1]=i
					if (i == pSize) {// ��i==���ݿ�Ĵ�С
						Arrays.fill(isFind, true);// isFind��Ϊȫ���ҵ���,!!����ȡ����������һ��ȫȡ��Ϊ������С
					} // Of if
				} else {
					isFind[j] = true;// ֻ��Ϊ��j���ҵ������Ž�
				} // Of if
			} // Of for j
				// System.out.println("QueriedInstance: " + i + ", cost: " +
				// tmpCost[0] + "\t");
			if (isFind[0] && isFind[1]) {// ����0��͵�1�඼�ҵ������Ÿ���
				Arrays.fill(isFind, false);// ȫ��д��false
				for (int j = 0; j <= i; j++) {// ������������֮ǰ�� ����и�С�ĸ���ȥ�ó����Ž⣬�Ǿͼ����������ٵĸ���
					for (int k = 0; k < 2; k++) {
						tmpCost[k] = (1 - ra[j]) * mCost[k] * pSize + tCost * j;
						if (tmpCost[k] <= tmpMinCost[k] && !isFind[k]) {
							isFind[k] = true;
							starLoose[k] = j;
						} // Of if
						if (isFind[0] && isFind[1]) {
							return starLoose;// {1,2}���ص�һ��͵ڶ���Ӧ�ó�ȡ�ĸ���������˵Ԥ��ĸ�������i���Ƚ����Ǿͳ�i�� ������С
						} // Of if
					} // Of for k
				} // Of for j
			} // Of if
		} // Of for i
		return new int[] { 0, 0 };// ��û���ҵ��ͷ���0.0��
		// throw new RuntimeException("Error occured in lookup("+pSize+")");
	}// Of lookup
}
