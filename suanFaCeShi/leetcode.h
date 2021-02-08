#pragma once
#include <string>
#include <vector>
#include<unordered_map>
using namespace std;
#ifndef leetcode_H
#define leetcode_H
class leetcode
{
public:
	leetcode();
	~leetcode();
	/*力扣算法题库第389题
	找不同*/	
	char findTheDifference(string s, string t);
	char findTheDifference2(string s, string t);

	/*Given a string containing just the characters ‘(‘, ‘)’, ‘{ ‘, ‘ }’, ‘[’
	and ‘]’, determine if the input string is valid.
	The brackets must close in the correct order, “()”and “()[] {}” are
	all valid but “(]”and “([)]” are not.*/

	bool isValid(string s);
	bool isVaild2(string s);


#pragma region 字节跳动 3.无重复字符的最长子串
	int lengthOfLongestSubstring(string s);//用哈希set

	int lengthOfLongestSubstring_byMap(string s);//用哈希表

	int lengthOfLongerstSubstring2(string s);//更易于理解的窗口实现形式
#pragma endregion

#pragma region 76.最小覆盖子串
	string minWindow(string s, string t);
#pragma endregion

#pragma region 509.斐波那契数列
	/*解法1：最传统的递归解法，效率比较低，
	浪费内存比较严重
	*/
	int fib_recursion(int n); 

   /*解法2：
   利用滚动数组，节约空间复杂度,时间复杂度为O(n)
   */
	int fib_gundong(int n);

   /*解法3：
    用通项公式求解 
   */

	int fib_tongxiang(int n);

#pragma endregion


#pragma region LCP08剧情触发时间
	/*
	维护一个递增的Increse数组，其每个元素都是严格递增的，对requirements里的每个元素 在Increse数组里二分查找到第一个比它大的，极为首先满足条件的天数

	作者：skhhh
	链接：
	来源：力扣（LeetCode）
	著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
	*/
	vector<int> getTriggerTime(vector<vector<int>>& increase, vector<vector<int>>& requirements);
#pragma endregion 




#pragma region 35搜索插入位置(考察二分查找)
	int searchInsert(vector<int>& nums, int target);
#pragma endregion

#pragma region 81搜索旋转排序数组 II
	bool search(vector<int>& nums, int target);
#pragma endregion

#pragma region 74.搜索二维矩阵
	/*解法1：我自己的思想
	将二维数组的下标转换为连续的一维数组下标*/
	bool searchMatrix(vector<vector<int>>& matrix, int target); 

	/*解法2：
	选择矩阵左下角作为起始元素 Q
	如果 Q > target，右方和下方的元素没有必要看了（相对于一维数组的右边元素）
	如果 Q < target，左方和上方的元素没有必要看了（相对于一维数组的左边元素）
	如果 Q == target ，直接 返回 True
	交回了都找不到，返回 False


	如何选出发点：
	选左上角，往右走和往下走都增大，不能选

	选右下角，往上走和往左走都减小，不能选

	选左下角，往右走增大，往上走减小，可选

	选右上角，往下走增大，往左走减小，可选

	*/
	bool searchMatrix2(vector<vector<int>>& matrix, int target);
#pragma endregion


#pragma region 240.搜索二维矩阵II
	bool searchMatrixEnhance(vector<vector<int>>& matrix, int target);
#pragma endregion

#pragma region 153.寻找旋转排序数组中的最小值
	int findMin(vector<int>& nums);


	int findMin2(vector<int>& nums);//自己的想法


	//leetcode题解详细注解版
	int findMin3(vector<int>& nums);


	//通过比较左边界来寻找最大值，最大值+1 就是最小值
	int finMax(vector<int> & nums);

	//如果循环条件是l<=r，该如何写
	int finMin4(vector<int> & nums);
#pragma endregion

#pragma region 1018. 可被5整除的二进制前缀
	vector<bool> prefixesDivBy5(vector<int>& A);
#pragma endregion


#pragma region 用位运算进行加减乘除

	int add(int a, int b);//递归形式实现加法
	
	
	
	//将负数的二进值转化为正数形式
	inline int negative(int i) {
		return add(~i, 1);
	}

	/*减法其实是用加法来实现的。在ALU中，当我求a-b，其实是求[a-b]补。因为有[a-b]补=[a]补 - [b]补= [a]补+[-b]补。
	所以我就要先求出-b。求一个数的负的操作是将其连符号位一起取反然后加1。
	*/
	int subTraction(int a, int b);////减法运算：利用求负操作和加法操作


	inline int getSign(int i) {
		return (i >> 31);
	}

	inline int bePositive(int i) {
		if (i >> 31) return  negative(i);
		else return i;
	}

	int multiply(int a, int b);

#pragma endregion 

#pragma region 154.寻找旋转排序数组中的最小值II
	int findMinDuplicate(vector<int>& nums);
#pragma endregion

#pragma region 875.爱吃香蕉的珂珂,二分法应用
	//判断在这个给定的K速度下能不能在H时间内吃完所有的香蕉
	bool possible(vector<int>& piles, int H, int K);

	int minEatingSpeed(vector<int>& piles, int H);
#pragma endregion


#pragma region 1011 在D天内送达包裹的能力，二分法应用
	int shipWithinDays(vector<int>& weights, int D);
	bool possible2(vector<int>& weights, int D, int C);
#pragma endregion


#pragma region 27.移除元素
	//解法1：双层循环暴力解法
	int removeElement(vector<int>& nums, int val);

	//解法2：双指针法
	int removeElementSzz(vector<int>& nums, int val);
#pragma endregion


#pragma region 628.三个数的最大乘积
	int maximumProduct(vector<int>& nums);

	/*解法2：线性扫描
	* 
	* 我们实际上只要求出数组中最大的三个数以及最小的两个数，因此我们可以不用排序，用线性扫描直接得出这五个数。
	*/
	int maximumProduct2(vector<int>& nums);
#pragma endregion

#pragma region 50.pow(x,n)
	double myPow(double x, int n);

	double quickMul(double x, long long N);


	/*解法2：快速幂+迭代
	* 每个二进制数位都有一个权值，权值如下图所示，最终结果就等于所有二进制位为1的权值之积，, 例如上述 x^77次方对应的二进制 (1001101) 和每个二进制位的权值如下

1	0	0	1	1	0	1
x^64	x^32	x^16	x^8	x^4	x^2	x^1

最终结果就是所有二进制位为1的权值之积：x^1 * x^4 * x^8 * x^64 = x^77
	*/
	double quickMul2(double x, long long N);
#pragma endregion

#pragma region 两数之和
	//使用哈希表
	vector<int> twoSum(vector<int>& nums, int target);

#pragma endregion


#pragma region 15.三数之和
	//哈斯法，双循环
	vector<vector<int>> threeSum(vector<int>& nums);

	//双指针法
	vector<vector<int>> threeSum2(vector<int>& nums);


	//双指针法 while
	vector<vector<int>> threeSum3(vector<int>& nums);
#pragma endregion

	//贪心算法
#pragma region 674 最长连续递增序列
	int findLengthOfLCIS(vector<int>& nums);
#pragma endregion


#pragma region 334.递增的三元子序列
	bool increasingTriplet(vector<int>& nums);
#pragma endregion

#pragma region 2.两数相加
	typedef struct ListNode {
		int val;
		ListNode* next;
		ListNode() : val(0), next(nullptr) {}
		ListNode(int x) : val(x), next(nullptr) {}
		ListNode(int x, ListNode* next) : val(x), next(next) {}
		
	}ListNode;
	ListNode* addTwoNumbers(ListNode* l1, ListNode* l2);

	//解法2：不使用额外链表，直接在原表上修改
	/*直接在一条链表上修改 省内存 代码简洁
	* https://leetcode-cn.com/problems/sum-lists-lcci/solution/zhi-jie-zai-yi-tiao-lian-biao-shang-xiu-gai-sheng-/
	*/
	ListNode* addTwoNumbers2(ListNode* l1, ListNode* l2);
#pragma endregion



#pragma region 424. 替换后的最长重复字符
	int characterReplacement(string s, int k);
#pragma endregion

#pragma region 461. 汉明距离
	int hammingDistance(int x, int y);
#pragma endregion

#pragma region 1208. 尽可能使字符串相等

	//解法1，前缀和加二分法
	int equalSubstring(string s, string t, int maxCost);

	int binarySearch(const vector<int>& accDiff, int endIndex, int target);
#pragma endregion	

#pragma region 4. 寻找两个正序数组的中位数
	//解法1，折半删除区间
	double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2);

	int getKthElement(const vector<int>& nums1, const vector<int>& nums2, int k);
#pragma endregion

#pragma region 1423. 可获得的最大点数
	//解法1，滑动窗口，让剩下的子数组取最小值，剩下的子数组的大小是固定的，用一个窗口滑
	int maxScore(vector<int>& cardPoints, int k);
#pragma endregion



#pragma region 209.长度最小的子数组
	int minSubArrayLen(int target, vector<int>& nums);//解法1，滑动数组



	int minSubArrayLen2(int target, vector<int>& nums);//解法2，前缀和+ 二分查找
#pragma endregion

#pragma region 7整数反转
	int reverse(int x);
#pragma endregion



#pragma region 9.回文数
	/*解法1，将整数反转后与原数值进行比较
	*/
	bool isPalindrome(int x);


	/*解法2：反转一半数字
	*/
	bool isPalindrome2(int x);
#pragma endregion

#pragma region 8. 字符串转换整数 (atoi)
	/*解法1：按普通的思维去构建代码
	*/
	int myAtoi(string s);
	

	//解法2 自动机
	class Automation {
	public:
		string state = "start";

		int sign = 1;
		long long ans = 0;
		/*
		*
			' '			+/-			number			other
start		start		signed		in_number		end
signed		end			end			in_number		end
in_number	end			end			in_number		end
end			end			end			end				end

		*/
		
		unordered_map<string, vector<string> > table = {
	 {"start", {"start", "signed", "in_number", "end"}},
	 {"signed", {"end", "end", "in_number", "end"}},
	 {"in_number", {"end", "end", "in_number", "end"}},
	 {"end", {"end", "end", "end", "end"}}
		};

	

		//根据传入的字符判断返回的状态
		int get_col(char c) {
			if (isspace(c)) return 0;
			if (c == '+' or c == '-')//c++也支持and or
			return 1;
			if (isdigit) return 2; //各种状态表示.0,1,2,3对应额是状态表里的列序号
			return 3;
		}

		

		void get(char c) {
			state = table[state][get_col(c)];
			if (state == "in_number") {
				ans = ans * 10 + c - '0';
				ans = sign == 1 ? min(ans, (long long)INT_MAX) : min(ans, -(long long)INT_MIN);
			}
			else if (state == "signed")
				sign = c == '+' ? 1 : -1;
		}
	};
	int myAtoi2(string s);
#pragma endregion


#pragma region 344. 反转字符串
	void reverseString(vector<char>& s);
#pragma endregion

#pragma region 5. 最长回文子串
	string longestPalindrome(string s);
#pragma endregion
private:

};



#endif
