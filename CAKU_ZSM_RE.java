package classification;

import java.io.FileReader;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import weka.core.*;

public class CAKU_ZSM_RE {
	int instanceStates[];
	int labels[];
	Instances data;
	static double teachCost;
	static double[] misclassifiedCost;

	/**
	 * *******************************
	 * Constructor of file
	 * @param fileName
	 * @param misclassifiedCost
	 * @param teachCost
	 * *******************************
	 */
	public CAKU_ZSM_RE(String fileName, double[] misclassifiedCost, double teachCost) {
		CAKU_ZSM_RE.teachCost = teachCost;
		CAKU_ZSM_RE.misclassifiedCost = misclassifiedCost;
		try {
			FileReader fileReader = new FileReader(fileName);
			data = new Instances(fileReader);
			fileReader.close();

		} catch (Exception exceptionOfFilereader) {
			// TODO: handle exception
			System.out.println("file could not read!");
		}
		instanceStates = new int[data.numInstances()];
		labels = new int[data.numInstances()];
		Arrays.fill(instanceStates, -1);
		Arrays.fill(labels, -1);

	}

	/**
	 * *******************************
	 * Distance between two point
	 * @param indexOfFirstPoint
	 * @param indexOfSecondPoint
	 * @return
	 * *******************************
	 */
	public double computeDistances(int indexOfFirstPoint, int indexOfSecondPoint) {
		double reOfdistance = 0;
		for (int i = 0; i < data.numAttributes() - 1; i++)
			reOfdistance += Math
					.pow(data.instance(indexOfFirstPoint).value(i) - data.instance(indexOfSecondPoint).value(i), 2);
		return reOfdistance;
	}

	/**
	 * *******************************
	 * The farthest point for all centers 
	 * @param paraBlock
	 * @param paraList
	 * @return
	 * *******************************
	 */
	int computeFathestDistance(int[] paraBlock, ArrayList<Integer> paraList) {
		int indexOfFarthestPoint = -1;
		int isFind[] = new int[paraBlock.length];
		Arrays.fill(isFind, 0);
		for (int i = 0; i < paraList.size(); i++) {
			isFind[paraList.get(i)] = 1;
		}
		double tempMaxDistance = 0;
		for (int i = 0; i < paraBlock.length; i++) {
			double tempDistance = 0;

			for (int j = 0; j < paraList.size(); j++) {
				tempDistance += computeDistances(paraBlock[i], paraBlock[paraList.get(j)]);
			}
			if (tempDistance > tempMaxDistance && isFind[i] != 1) {
				tempMaxDistance = tempDistance;
				indexOfFarthestPoint = i;
			}
		}
		return indexOfFarthestPoint;
	}

	/**
	 * Mark all data
	 */
	void markData() {
		int[] tempAlldataIndex = new int[data.numInstances()];
		for (int i = 0; i < data.numInstances(); i++)
			tempAlldataIndex[i] = i;

		qurriedAndClassify(tempAlldataIndex);
		System.out.println("instanceStates: " + Arrays.toString(instanceStates));
		System.out.println("labels: " + Arrays.toString(labels));
	}

	/**
	 * *******************************
	 * Begin learn paraBlock
	 * @param paraBlock:The data need to learn
	 * *******************************
	 */
	void qurriedAndClassify(int[] paraBlock) {
		int indexOfFirstLabel = -1;
		int indexOfSecondLabel = -1;
		ArrayList<Integer> presentivePoint = new ArrayList<>();

		//Step 1.Find the point existed label.  
		for (int i = 0; i < paraBlock.length; i++)
			if (labels[paraBlock[i]] != -1) {
				indexOfFirstLabel = i;
				break;
			}

		//Step 2.Find other point with label.
		//		 If label is same as firstlabel,add to List.
		//		 otherwise,split.
		if (indexOfFirstLabel != -1) {
			presentivePoint.add(indexOfFirstLabel);
			for (int i = indexOfFirstLabel + 1; i < paraBlock.length; i++) {
				if (labels[paraBlock[i]] == 1) {
					indexOfSecondLabel = i;
					if (labels[paraBlock[indexOfFirstLabel]] != labels[paraBlock[indexOfSecondLabel]]) {
						// System.out.println("Now split.");
						System.out.println("oneBlocktwoLabels" + paraBlock[indexOfSecondLabel]);
						splitParablock(paraBlock, indexOfSecondLabel, indexOfFirstLabel);
						return;
					} else {
						presentivePoint.add(indexOfSecondLabel);
					}
				}
			}
		}

		if (indexOfFirstLabel == -1) {
			presentivePoint.add(0);
			instanceStates[paraBlock[presentivePoint.get(0)]] = 1;
			labels[paraBlock[presentivePoint.get(0)]] = (int) data.instance(paraBlock[presentivePoint.get(0)])
					.value(data.numAttributes() - 1);
		}
		
		//Step 3.Compute the number of need to buy.
		int numOfpresentatives = findRepresentatives(paraBlock, labels[presentivePoint.get(0)]);
		
		//Step 4.Find the same label and farthest point until full.
		while (presentivePoint.size() < numOfpresentatives) {
			int farthestPointIndex = computeFathestDistance(paraBlock, presentivePoint);
			instanceStates[paraBlock[farthestPointIndex]] = 1;
			labels[paraBlock[farthestPointIndex]] = (int) data.instance(paraBlock[farthestPointIndex])
					.value(data.numAttributes() - 1);
			if (labels[paraBlock[farthestPointIndex]] == labels[paraBlock[presentivePoint.get(0)]]) {
				presentivePoint.add(farthestPointIndex);
				continue;
			} else {
				System.out.println("differentLabelIndex" + paraBlock[farthestPointIndex]);
				splitParablock(paraBlock, presentivePoint.get(0), farthestPointIndex);
				return;
			}
		}
		
		//Step 5.Predict other data.
		for (int i = 0; i < paraBlock.length; i++) {
			if (instanceStates[paraBlock[i]] != 1) {
				instanceStates[paraBlock[i]] = 2;
				labels[paraBlock[i]] = labels[paraBlock[presentivePoint.get(0)]];
			}
		}
	}

	/**
	 * *******************************
	 * Compute distance between two points
	 * @param indexOfPoint
	 * @param center:one center's attributes
	 * @return Distance
	 * *******************************
	 */
	double computeDistanceBetweenTwoPoint(int indexOfPoint, double[] center) {
		double resultOfDistance = 0;
		for (int i = 0; i < data.numAttributes() - 1; i++) {
			resultOfDistance += Math.abs(data.instance(indexOfPoint).value(i) - center[i]);
		}
		return resultOfDistance;
	}

	/**
	 * *******************************
	 * Computer two matrix is same?
	 * @param matrix1
	 * @param matrix2
	 * @return
	 * *******************************
	 */
	boolean MatrixIsEqual(double[][] matrix1, double[][] matrix2) {
		for (int i = 0; i < matrix1.length; i++) {
			for (int j = 0; j < matrix1[0].length; j++) {
				if (Math.abs(matrix1[i][j] - matrix2[i][j]) > 1e-6) {
					return false;
				} else {
					continue;
				}
			}
		}
		return true;
	}

	/**
	 * *******************************
	 * 2Means cluster method
	 * @param paraBlock
	 * @param indexOfOne
	 * @param indexOfTwo
	 * @return
	 * *******************************
	 */
	int[] kMeansFarDistance(int[] paraBlock, int indexOfOne, int indexOfTwo) {
		double[][] currentCenter = new double[2][data.numAttributes() - 1];
		double[][] newCenter = new double[2][data.numAttributes() - 1];
		int[] resultOfClassify = new int[paraBlock.length];
		int[] tempResultOfClassify = new int[paraBlock.length];
		for (int i = 0; i < data.numAttributes() - 1; i++) {
			newCenter[0][i] = data.instance(paraBlock[indexOfOne]).value(i);
			newCenter[1][i] = data.instance(paraBlock[indexOfTwo]).value(i);
		}

		while (!MatrixIsEqual(currentCenter, newCenter)) {
			currentCenter = newCenter;
			for (int i = 0; i < paraBlock.length; i++) {
				double tempDistanceNear = Double.MAX_VALUE;
				for (int j = 0; j < 2; j++) {
					double tempDistance = 0;
					tempDistance += computeDistanceBetweenTwoPoint(paraBlock[i], currentCenter[j]);
					if (tempDistance < tempDistanceNear) {
						tempResultOfClassify[i] = j;
						tempDistanceNear = tempDistance;
					}
				}
			}
			int[] tempCount = new int[2];
			for (int i = 0; i < paraBlock.length; i++) {
				tempCount[tempResultOfClassify[i]]++;
			}

			newCenter = new double[2][data.numAttributes() - 1];

			for (int i = 0; i < paraBlock.length; i++) {
				for (int j = 0; j < data.numAttributes() - 1; j++)
					newCenter[tempResultOfClassify[i]][j] += data.instance(paraBlock[i]).value(j);
			}
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < data.numAttributes() - 1; j++) {
					newCenter[i][j] = newCenter[i][j] / tempCount[i];
				}
			}

		}
		resultOfClassify = tempResultOfClassify;
		return resultOfClassify;
	}

	/**
	 * *******************************
	 * Split block to two
	 * @param paraBlock
	 * @param indexOfOne
	 * @param indexOfTwo
	 * *******************************
	 */
	void splitParablock(int[] paraBlock, int indexOfOne, int indexOfTwo) {
		int[] cluster = kMeansFarDistance(paraBlock, indexOfOne, indexOfTwo);
		int[] count = new int[2];
		for (int i = 0; i < paraBlock.length; i++)
			count[cluster[i]]++;

		int[][] clusifiedOfAll = new int[2][];
		clusifiedOfAll[0] = new int[count[0]];
		clusifiedOfAll[1] = new int[count[1]];
		Arrays.fill(count, 0);
		for (int i = 0; i < paraBlock.length; i++) {
			clusifiedOfAll[cluster[i]][count[cluster[i]]++] = paraBlock[i];
		}
		qurriedAndClassify(clusifiedOfAll[0]);
		qurriedAndClassify(clusifiedOfAll[1]);
	}

	/**
	 * *******************************
	 * Total cost
	 * @return
	 * *******************************
	 */
	double TTCost() {
		double cost = 0;
		double teachCostNumber = 0;
		double misClassCost = 0;

		for (int i = 0; i < data.numInstances(); i++) {
			if (instanceStates[i] == 1) {
				cost += teachCost;
				teachCostNumber++;
			} else {
				if (labels[i] == 0 && (int) data.instance(i).value(data.numAttributes() - 1) == 1) {
					cost += misclassifiedCost[0];
					misClassCost += misclassifiedCost[0];
				} else if (labels[i] == 1 && (int) data.instance(i).value(data.numAttributes() - 1) == 0) {
					cost += misclassifiedCost[1];
					misClassCost += misclassifiedCost[1];
				}
			}
		}

		System.out.println("teachCost(numbers):" + teachCostNumber);
		System.out.println("misclassifiedCost:" + misClassCost);
		return cost;
	}

	/**
	 * *******************************
	 * Main method
	 * @param args
	 * *******************************
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		double[] misCost = { 2, 4 };
		double teachCost = 1;
		double totalCost = 0;
		CAKU_ZSM_RE tempClassified = new CAKU_ZSM_RE("data/iris2classes.arff", misCost, teachCost);
		tempClassified.markData();
		totalCost = tempClassified.TTCost();
		System.out.println(totalCost);
	}

	public static double expectPosNum(int R, int B, int N) {
		BigDecimal fenzi = new BigDecimal("0");
		BigDecimal fenmu = new BigDecimal("0");
		for (int i = R; i <= N - B; i++) {
			BigDecimal a = A(R, i).multiply(A(B, N - i));
			fenzi = fenzi.add(a.multiply(new BigDecimal("" + i)));
			fenmu = fenmu.add(a);
		}
		return fenzi.divide(fenmu, 4, BigDecimal.ROUND_HALF_EVEN).doubleValue();

	}// Of expectPosNum

	private int[] lookup(int pSize) {
		double[] tmpMinCost = new double[] { 0.5 * pSize * misclassifiedCost[0], 0.5 * pSize * misclassifiedCost[1] };
		int[] star = new int[2];
		int[] starLoose = new int[2];
		boolean[] isFind = new boolean[2];
		double[] ra = new double[pSize + 1];
		// ra[]������ǳ鵽��i��ʱ���ж�������Ϊ�������Ǵ���
		ra[0] = 0.5;// ��һ������Ǻ���������ĸ��ʶ���0.5�����Ұ��������е��뷨��������Ķ���һ�µ���ɫ
		double[] tmpCost = new double[2];// 1���Ϊ2��Ĵ����2���Ϊ1��Ĵ�����󶼼���teachCost��
		for (int i = 1; i <= pSize; i++) {
			if (pSize >= 1000) {
				ra[i] = (i + 1.0) / (i + 2.0);// ���� �����ȴ���1000�������鵽��i����ͬ��ɫʱ�򴿵ĸ�����i+1/i+2
			} else {
				ra[i] = expectPosNum(i, 0, pSize) / pSize;// ��С��1000�����ֱ�Ӽ��������ı��� ����101������ ����0-100����һ�����
															// ��ÿ������������˸��ʣ���ͣ�����80����80-100�ĺͣ����������101 ���������ĸ���
			} // ����ÿ�������ĸ����¿��ܵĴ��ۡ�
			for (int j = 0; j < 2; j++) {// 1���Ϊ2��Ĵ����2���Ϊ1��Ĵ�����󶼼���teachCost��
				tmpCost[j] = (1 - ra[i]) * misclassifiedCost[j] * pSize + teachCost * i;// 0����1�������Ĵ���
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
						tmpCost[k] = (1 - ra[j]) * misclassifiedCost[k] * pSize + teachCost * j;
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
			tempBuyLabels = tempNumBuys[tempFirstLabel];
			// ��������������ĸ�����ȥ�Ѿ���ĸ���
		} else {
			tempBuyLabels = Math.max(tempNumBuys[0], tempNumBuys[1]);
			// �������������lookup���ص����ֵ��ȥ�Ѿ���ĸ�����
		} // Of if

		// int tempBuyLabels = pureThreshold(paraBlock.length) - tempLabels;
		// System.out.println("number to buy:" + tempBuyLabels);
		System.out.println("need buy" + tempBuyLabels);
		return tempBuyLabels;
	}// Of findRepresentatives

	public static BigDecimal A(int m, int n) {
		if (m > n) {
			return new BigDecimal("0");
		} // Of if
		BigDecimal re = new BigDecimal("1");
		for (int i = n - m + 1; i <= n; i++) {
			re = re.multiply(new BigDecimal(i));
		} // Of if
		return re;
	}
}
