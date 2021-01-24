#pragma once
#include <string>
#include <vector>
using namespace std;
#ifndef leetcode_H
#define leetcode_H
class leetcode
{
public:
	leetcode();
	~leetcode();
	/*�����㷨����389��
	�Ҳ�ͬ*/	
	char findTheDifference(string s, string t);
	char findTheDifference2(string s, string t);

	/*Given a string containing just the characters ��(��, ��)��, ��{ ��, �� }��, ��[��
	and ��]��, determine if the input string is valid.
	The brackets must close in the correct order, ��()��and ��()[] {}�� are
	all valid but ��(]��and ��([)]�� are not.*/

	bool isValid(string s);
	bool isVaild2(string s);


#pragma region �ֽ����� 3.���ظ��ַ�����Ӵ�
	int lengthOfLongestSubstring(string s);//�ù�ϣset

	int lengthOfLongestSubstring_byMap(string s);//�ù�ϣ��

	int lengthOfLongerstSubstring2(string s);//���������Ĵ���ʵ����ʽ
#pragma endregion

#pragma region 76.��С�����Ӵ�
	string minWindow(string s, string t);
#pragma endregion

#pragma region 509.쳲���������
	/*�ⷨ1���ͳ�ĵݹ�ⷨ��Ч�ʱȽϵͣ�
	�˷��ڴ�Ƚ�����
	*/
	int fib_recursion(int n); 

   /*�ⷨ2��
   ���ù������飬��Լ�ռ临�Ӷ�,ʱ�临�Ӷ�ΪO(n)
   */
	int fib_gundong(int n);

   /*�ⷨ3��
    ��ͨ�ʽ��� 
   */

	int fib_tongxiang(int n);

#pragma endregion


#pragma region LCP08���鴥��ʱ��
	/*
	ά��һ��������Increse���飬��ÿ��Ԫ�ض����ϸ�����ģ���requirements���ÿ��Ԫ�� ��Increse��������ֲ��ҵ���һ��������ģ���Ϊ������������������

	���ߣ�skhhh
	���ӣ�
	��Դ�����ۣ�LeetCode��
	����Ȩ���������С���ҵת������ϵ���߻����Ȩ������ҵת����ע��������
	*/
	vector<int> getTriggerTime(vector<vector<int>>& increase, vector<vector<int>>& requirements);
#pragma endregion 




#pragma region 35��������λ��(������ֲ���)
	int searchInsert(vector<int>& nums, int target);
#pragma endregion

#pragma region 81������ת�������� II
	bool search(vector<int>& nums, int target);
#pragma endregion

#pragma region 74.������ά����
	/*�ⷨ1�����Լ���˼��
	����ά������±�ת��Ϊ������һά�����±�*/
	bool searchMatrix(vector<vector<int>>& matrix, int target); 

	/*�ⷨ2��
	ѡ��������½���Ϊ��ʼԪ�� Q
	��� Q > target���ҷ����·���Ԫ��û�б�Ҫ���ˣ������һά������ұ�Ԫ�أ�
	��� Q < target���󷽺��Ϸ���Ԫ��û�б�Ҫ���ˣ������һά��������Ԫ�أ�
	��� Q == target ��ֱ�� ���� True
	�����˶��Ҳ��������� False


	���ѡ�����㣺
	ѡ���Ͻǣ������ߺ������߶����󣬲���ѡ

	ѡ���½ǣ������ߺ������߶���С������ѡ

	ѡ���½ǣ����������������߼�С����ѡ

	ѡ���Ͻǣ����������������߼�С����ѡ

	*/
	bool searchMatrix2(vector<vector<int>>& matrix, int target);
#pragma endregion


#pragma region 240.������ά����II
	bool searchMatrixEnhance(vector<vector<int>>& matrix, int target);
#pragma endregion

#pragma region 153.Ѱ����ת���������е���Сֵ
	int findMin(vector<int>& nums);


	int findMin2(vector<int>& nums);//�Լ����뷨


	//leetcode�����ϸע���
	int findMin3(vector<int>& nums);


	//ͨ���Ƚ���߽���Ѱ�����ֵ�����ֵ+1 ������Сֵ
	int finMax(vector<int> & nums);

	//���ѭ��������l<=r�������д
	int finMin4(vector<int> & nums);
#pragma endregion

#pragma region 1018. �ɱ�5�����Ķ�����ǰ׺
	vector<bool> prefixesDivBy5(vector<int>& A);
#pragma endregion


#pragma region ��λ������мӼ��˳�

	int add(int a, int b);//�ݹ���ʽʵ�ּӷ�
	
	
	
	//�������Ķ���ֵת��Ϊ������ʽ
	inline int negative(int i) {
		return add(~i, 1);
	}

	/*������ʵ���üӷ���ʵ�ֵġ���ALU�У�������a-b����ʵ����[a-b]������Ϊ��[a-b]��=[a]�� - [b]��= [a]��+[-b]����
	�����Ҿ�Ҫ�����-b����һ�����ĸ��Ĳ����ǽ���������λһ��ȡ��Ȼ���1��
	*/
	int subTraction(int a, int b);////�������㣺�����󸺲����ͼӷ�����


	inline int getSign(int i) {
		return (i >> 31);
	}

	inline int bePositive(int i) {
		if (i >> 31) return  negative(i);
		else return i;
	}

	int multiply(int a, int b);

#pragma endregion 

#pragma region 154.Ѱ����ת���������е���СֵII
	int findMinDuplicate(vector<int>& nums);
#pragma endregion

#pragma region 875.�����㽶������,���ַ�Ӧ��
	//�ж������������K�ٶ����ܲ�����Hʱ���ڳ������е��㽶
	bool possible(vector<int>& piles, int H, int K);

	int minEatingSpeed(vector<int>& piles, int H);
#pragma endregion


#pragma region 1011 ��D�����ʹ���������������ַ�Ӧ��
	int shipWithinDays(vector<int>& weights, int D);
	bool possible2(vector<int>& weights, int D, int C);
#pragma endregion


#pragma region 27.�Ƴ�Ԫ��
	//�ⷨ1��˫��ѭ�������ⷨ
	int removeElement(vector<int>& nums, int val);

	//�ⷨ2��˫ָ�뷨
	int removeElementSzz(vector<int>& nums, int val);
#pragma endregion


#pragma region 628.�����������˻�
	int maximumProduct(vector<int>& nums);

	/*�ⷨ2������ɨ��
	* 
	* ����ʵ����ֻҪ��������������������Լ���С����������������ǿ��Բ�������������ɨ��ֱ�ӵó����������
	*/
	int maximumProduct2(vector<int>& nums);
#pragma endregion

#pragma region 50.pow(x,n)
	double myPow(double x, int n);

	double quickMul(double x, long long N);


	/*�ⷨ2��������+����
	* ÿ����������λ����һ��Ȩֵ��Ȩֵ����ͼ��ʾ�����ս���͵������ж�����λΪ1��Ȩֵ֮����, �������� x^77�η���Ӧ�Ķ����� (1001101) ��ÿ��������λ��Ȩֵ����

1	0	0	1	1	0	1
x^64	x^32	x^16	x^8	x^4	x^2	x^1

���ս���������ж�����λΪ1��Ȩֵ֮����x^1 * x^4 * x^8 * x^64 = x^77
	*/
	double quickMul2(double x, long long N);
#pragma endregion

#pragma region ����֮��
	//ʹ�ù�ϣ��
	vector<int> twoSum(vector<int>& nums, int target);

#pragma endregion


#pragma region 15.����֮��
	//��˹����˫ѭ��
	vector<vector<int>> threeSum(vector<int>& nums);

	//˫ָ�뷨
	vector<vector<int>> threeSum2(vector<int>& nums);


	//˫ָ�뷨 while
	vector<vector<int>> threeSum3(vector<int>& nums);
#pragma endregion

	//̰���㷨
#pragma region 674 �������������
	int findLengthOfLCIS(vector<int>& nums);
#pragma endregion


#pragma region 334.��������Ԫ������
	bool increasingTriplet(vector<int>& nums);
#pragma endregion


private:

};



#endif
