#include "leetcode.h"
#include<string>
#include <stack>
#include<unordered_set>
#include<math.h>
#include<algorithm>
#include<unordered_map>
using namespace std;


leetcode::leetcode()
{

}
leetcode::~leetcode()
{

}
#pragma region 力扣算法题库第389题

//解法1，利用异或运算，利用了a^b^c^a^b^c^d =d的兴致。
//注意要讲字符串转化为字符*/
char leetcode::findTheDifference(string s, string t) {
	int ret = 0;
	for (char ch : s) {  //写成 char &ch :s 可以别原写法节约内存，原因是&ch直接在原字符串s上进行遍历，原写法要复制出个s再进行遍历操作
		ret ^= ch;
	}
	for (char ch : t) {
		ret ^= ch;
	}
	return ret;
}

////解法2，利用将每个字符的ascii码求和，然后用字符串1的
////和减去字符串2的和既代表了被添加的字符

char leetcode::findTheDifference2(string s, string t) {
	int ret1 = 0;
	int ret2 = 0;
	for (char ch : s) {
		ret1 += ch;
	}
	for (char ch : t) {
		ret2 += ch;
	}
	return abs( ret2 - ret1);
}

#pragma endregion

#pragma region 括号匹配问题
//给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效   有效字符串需满足：左括号必须用相同类型的右括号闭合。左括号必须以正确的顺序闭合。注意空字符串可被认为是有效字符串
bool leetcode::isValid(string s) {
	if (s.length() % 2 != 0)// 可以用s.length() & 1 == 1 来判断长度是否为偶数，位运算执行效率高
		return false;
	if (s.length() == 0)
		return true;
	stack<char> mystack;
	int i = 0;
	mystack.push('#');//哨兵元素，在数据结构中用于判断边界，减少遍历的开销
	while (i < s.size()) {
		if (s[i] == '(' || s[i] == '[' || s[i] == '{') {
			mystack.push(s[i]);
			++i;
		}
		else if ((s[i] == ')' && mystack.top() == '(') || (s[i] == ']' && mystack.top() == '[') || (s[i] == '}' && mystack.top() == '{'))
		{
			 ++i;
			mystack.pop();
		}
		else
			return false;
	}

	if (mystack.top() == '#')
		return true;
		return false;
}

bool leetcode::isVaild2(string s) {
	stack<int> st;
	for (int i = 0; i < s.size(); i++) {
		if (s[i] == '(') st.push(')');
		else if (s[i] == '{') st.push('}');
		else if (s[i] == '[') st.push(']');
		// 第三种情况 是遍历字符串匹配的过程中，栈已经为空了，没有匹配的字符了，说明右括号没有找到对应的左括号 return false
		// 第二种情况 遍历字符串匹配的过程中，发现栈里没有我们要匹配的字符。所以return false
		else if (st.empty() || st.top() != s[i]) return false;
		else st.pop(); // st.top() == s[i]
	}
	// 第一种情况 此时我们已经遍历完了字符串，但是栈不为空，说明有相应的左括号没有右括号来匹配，所以return false，否则就return true
	return st.empty();
}

#pragma endregion

#pragma region 字节跳动3.无重复字符的最长子串
int leetcode::lengthOfLongestSubstring(string s)
{
	//哈希集合，记录每个字符是否出现过
	unordered_set<char>occ;
	int n = s.size();
	//右指针，初始值为-1，相当于我们还在字符串的左侧边界，还没有开始移动
	int rk = -1, ans = 0;
	//枚举左指针的位置，初始值隐性表现为-1
	for (int i = 0; i < n; ++i) {
		if (i!= 0) {
			//左指针向右移动一格，移除一个字符
			occ.erase(s[i - 1]);
		}
		while (rk + 1 < n && !occ.count(s[rk + 1])){
			//不断移动右指针
			occ.insert(s[rk + 1]);
			++rk;
		}
		//第i到第rk个字符是一个极长的无重复字符子串
		ans = max(ans, rk - i + 1);
	}
	return ans;
}
int leetcode::lengthOfLongestSubstring_byMap(string s)
{
	if (s == "" || s.length() == 0) {
		return 0;
	}
	int ans = 0;
	int len = s.length();
	int start = 0;
	int end = 0;
	unordered_map<char, int>map;
	while (end < len) {
		ans = max(ans, end - start);
		// 当遇到重复值，说明左指针需要跳转，跳转的位置是该重复值的下标+1
		// 比如字符串abcdecf，到遇到第二个c，即便从bcde任意一个开始，长度都无法超过a，只有从decf开始计算才是新一轮查找
		// 值得注意的是，如果碰到了重复值的下标比左指针还小的情况，不应该跳转，因为左指针左边的元素不再窗口内，比如abba
		if (map.count(s[end]) && map.find(s[end])->second >= start) {
			start = map.find(s[end])->second + 1;
			map[s[end]] = end;//c++ 只能这样更新valUe的值
		}
		map.insert(pair<char, int>(s[end], end));// 无论重不重复都需要更新，该元素最近的下标
		end++;
	}
	return ans = max(ans, end - start);
}
int leetcode::lengthOfLongerstSubstring2(string s)
{
	if (s.size() == 0)
		return 0;
	unordered_set<char> lookUp;
	int maxStr = 0, left = 0;
	for (int i = 0; i < s.size(); i++) {
		while (lookUp.find(s[i]) != lookUp.end())
		{
			lookUp.erase(s[left]);
			left++;
		}
		maxStr = max(maxStr, i - left + 1);
		lookUp.insert(s[i]);
	}
	return maxStr;
}
#pragma endregion

#pragma region 76.最小覆盖子串
string leetcode::minWindow(string s, string t)
{
	if (s.size() == 0 || t.size() == 0)
		return "请输入字符";
	unordered_map<char, int>lookUp, minSubstring;

	return string();
}
#pragma endregion




#pragma region 斐波那契数列
int leetcode::fib_recursion(int n)
{
	if (n < 2) return n;

	return fib_recursion(n - 1) + fib_recursion(n - 2);
}

int leetcode::fib_gundong(int n)
{
	if (n < 2) return n;
	int p = 0, q = 0, r = 1;
	for (int i = 2; i <= n; ++i) {
		p = q;
		q = r;
		r = p + q;
	}
	return r;
}

int leetcode::fib_tongxiang(int n)
{
	double sqrt5 = sqrt(5);
	double fibN = pow((1 + sqrt5) / 2, n) - pow((1 - sqrt5) / 2, n);
	return round(fibN / sqrt(5));
}
#pragma endregion

//剧情触发
vector<int> leetcode::getTriggerTime(vector<vector<int> >& increase, vector<vector<int> >& requirements)
{
	vector<vector<int> > s(increase.size() + 1, vector<int>(3, 0));//vector<vector<int> > 最好空一格，否则有的编译器会报错
	for (int i = 0; i < increase.size(); i++) {
		for (int j = 0; j < 3; j++) {
			s[i + 1][j] = s[i][j] + increase[i][j];
		}
	}
	vector<int> ans;
	for (auto v : requirements) {
		int l = 0, r = increase.size();  // 二分查找
		while (l < r) {
			int m = (l + r) / 2;
			if (s[m][0] >= v[0] && s[m][1] >= v[1] && s[m][2] >= v[2])
				r = m;
			else
				l = m + 1;
		}
		if (s[l][0] >= v[0] && s[l][1] >= v[1] && s[l][2] >= v[2])
			ans.push_back(l);
		else
			ans.push_back(-1);
	}
	return ans;
}



//搜索插入位置
int leetcode::searchInsert(vector<int>& nums, int target)
{
	if (nums.empty()) return NULL;
	int l = 0, r = nums.size() - 1;//左右指针
	while (l<r)
	{
		int mid = l+((r-l)>>1);//最优中点取法
		if (nums[mid] >= target)
			r = mid;
		else
			l = mid + 1;
	}
	if (nums[l] < target) return l + 1;
	else return l;//循环结束时 l = r,所以返回哪个都一样
}

//81. 搜索旋转排序数组 II
bool leetcode::search(vector<int>& nums, int target)
{
	if (nums.empty()) return false;
	int l = 0, r = nums.size() - 1;
	while (l <= r) {
		int mid = l + ((r - l) >> 1);
		if (nums[mid] == target) {
			return true;
		}
		while (l<mid && nums[l] == nums[mid])
		{
			l += 1;
		}
		if (nums[l] <= nums[mid]) {
			if (nums[l] <= target && target < nums[mid]) {
				r = mid - 1;
			}
			else
				l = mid + 1;

		}
		else {
			if (nums[mid] < target && target <= nums[r]) {
				l = mid + 1;
			}
			else
				r = mid - 1;
		}
	 
	}
	return false;
}

//74.搜索二维矩阵
bool leetcode::searchMatrix(vector<vector<int>>& matrix, int target)
{
	if(matrix.empty() || target < -10000 || target > 10000)
	return false;
	int m = matrix.size(),n = matrix[0].size();
	int l = 0, r = m * n - 1;
	while (l <= r) {
		int mid = l + ((r - l) >> 1);
		int midH = mid / n, midV = mid % n;
		if (matrix[midH][midV] == target)
			return true;
		if (matrix[midH][midV] < target) {
			l = mid + 1;
		}
		else if(matrix[midH][midV] > target) {
			r = mid - 1;
		}
	}
	return false;
}

//西法的刷题秘籍解法
bool leetcode::searchMatrix2(vector<vector<int>>& matrix, int target)
{
	if (matrix.empty() || target < -10000 || target > 10000)
		return false;
	int m = matrix.size(), n = matrix[0].size();
	int x = m - 1, y = 0;
	while (x >= 0 && y < n) {
		if (matrix[x][y] < target) {
			y += 1;
		}
		else if (matrix[x][y] > target) {
			x -= 1;
		}
		else
			return true;
	}
	return false;
}

//240.搜索二维矩阵II
bool leetcode::searchMatrixEnhance(vector<vector<int>>& matrix, int target)
{
	int m = matrix.size(), n = matrix[0].size();
	int x = m - 1, y = 0;
	while (x >= 0 && y < n) {
		if (matrix[x][y] < target) {
			y += 1;
		}
		else if (matrix[x][y] > target) {
			x -= 1;
		}
		else
			return true;
	}
	return false;
}

//	53. 寻找旋转排序数组中的最小值
#pragma region
int leetcode::findMin(vector<int>& nums)
{
	if (nums.empty()) return -1;
	int l = 0, r = nums.size() - 1;
	while (l < r || nums[l] < nums[r]) {
		int mid = l + ((r - l) >> 1);
		if (nums[r] < nums[mid]) {
			l = mid +1 ;
		}
		else
		{
			r = mid ;
		}
	}
	return nums[l];
}

int leetcode::findMin2(vector<int>& nums)
{
	if (nums.empty()) return -1;
	int l = 0, r = nums.size() - 1;
	while (l <= r) {
		int mid = l + ((r - l) >> 1);
		if (nums[mid] > nums[mid + 1]) {
			return nums[mid + 1];
		}
		if (nums[mid - 1] > nums[mid]) {
			return nums[mid];
		}
		if (nums[mid] > nums[0]) {
			l = mid + 1;
		}
		else {
			r = mid - 1;
		}
	}
	return -1;
}

int leetcode::finMax(vector<int>& nums)
{
	int l = 0, r = nums.size() - 1;
	while (l < r) {
		int mid = l + ((r+1 - l) >> 1); /* 先加一再除，mid更靠近右边的right */
		if (nums[mid] > nums[l]) {
			l = mid;
		}
		else
			if (nums[mid] < nums[l]) {
				r = mid - 1;
			}
	}
	return nums[(r + 1) % nums.size()];/* 最大值向右移动一位就是最小值了（需要考虑最大值在最右边的情况，右移一位后对数组长度取余） */
}

int leetcode::finMin4(vector<int>& nums)
{
	int l = 0, r = nums.size() - 1;
	while (l <= r) { // 循环的条件选为左闭右闭区间left <= right
		int mid = l + ((r - l) >> 1);
		if (nums[mid] >= nums[r]) {// 注意是当中值大于等于右值时，
			l = mid + 1; // 将左边界移动到中值的右边
		}
		else { // 当中值小于右值时
			r = mid;      // 将右边界移动到中值处
		}
	}

	return nums[r];      // 最小值返回nums[right]
}



int leetcode::findMin3(vector<int>& nums) {
	int left = 0;
	int right = nums.size() - 1;                /* 左闭右闭区间，如果用右开区间则不方便判断右值 */
	while (left < right) {                      /* 循环不变式，如果left == right，则循环结束 */
		int mid = left + (right - left) / 2;    /* 地板除，mid更靠近left */
		if (nums[mid] > nums[right]) {          /* 中值 > 右值，最小值在右半边，收缩左边界 */
			left = mid + 1;                     /* 因为中值 > 右值，中值肯定不是最小值，左边界可以跨过mid */
		}
		else if (nums[mid] < nums[right]) {   /* 明确中值 < 右值，最小值在左半边，收缩右边界 */
			right = mid;                        /* 因为中值 < 右值，中值也可能是最小值，右边界只能取到mid处 */
		}
	}
	return nums[left];    /* 循环结束，left == right，最小值输出nums[left]或nums[right]均可 */
}
#pragma endregion

//1018.可被5整除的二进制数
vector<bool> leetcode::prefixesDivBy5(vector<int>& A)
{
	vector<bool> ans;
	int subResult = 0;
	int length = A.size();
	for (int &i : A) {
		subResult = ((subResult << 1) + i) % 5;
		ans.emplace_back(subResult == 0);
	}
	return ans;
}

#pragma region 位运算实现加减乘除
int leetcode::add(int a, int b)
{
	if (b == 0) //递归结束条件：如果右加数为0，即不再有进位了，则结束。
		return a;
	int s = a^b;
	int c = (a&b) << 1; //进位左移1位，达到进位的目的。
	return add(s, c); //再把'和'和'进位'相加。递归实现。
}
int leetcode::subTraction(int a, int b)
{
	return add(a,negative(b));
}


//很直观，就是用循环加法替代乘法。a*b，就是把a累加b次。时间复杂度为O(N)。
int leetcode::multiply(int a, int b)
{
	bool flag = true;
	if (getSign(a) == getSign(b)) {//积的符号判定
		flag = false;
	}
	a = bePositive(a);//先把乘数和被乘数变为正数
	b = bePositive(b);
	int ans = 0;
	while (b) {
		ans = add(ans, a);
		b = subTraction(b, 1);
	}
	if (flag)
		ans = negative(ans);
	return ans;
}
#pragma endregion


int leetcode::findMinDuplicate(vector<int>& nums)
{
	if (nums.empty()) return -1;
	int l = 0, r = nums.size() - 1;
	while (l < r) {
		int mid = l + ((r - l) >> 1);
		while (mid < r && nums[mid] == nums[r]) {
			r = r - 1;
		}
		if (nums[mid] <= nums[r])
			r = mid;
		else {
			l = l + 1;
		}
	}
	return nums[l];
}

bool leetcode::possible(vector<int>& piles, int H, int K)
{
	int sum = 0;
	for (int &i : piles) {
		double d = (double)i /(double) K; //整数相除向上取整也可以写成（i -1)/K + 1;
		sum += ceil(d);
	}
	
	return (sum > H) ? false: true;
}

int leetcode::minEatingSpeed(vector<int>& piles, int H)
{
	int l = 1;
	auto r = max_element(piles.begin(), piles.end());//此时r这个指针是指向容器中的最大元素
	int rr = (int)*r;
	while (l < rr) {//返回的是迭代器元素，要用*取出其中存的值
		int mid = l + ((rr - l) >> 1);
		if (!possible(piles, H, mid)) {
			l = mid + 1;
		}
		else
			rr = mid;//如果写成*r = mid,容器中的最大值也会被修改
	}
	return l;
}
bool leetcode::possible2(vector<int>& weights, int D, int C) {
	int sumD = 0, sumW = 0;
	for (int &i : weights) {
		sumW += i;
		if (sumW > C) {
			sumD += 1;
			sumW = i;
		}
	}
	return (sumD + 1) <= D;
}

int leetcode::shipWithinDays(vector<int>& weights, int D)
{
	int l = (int)*max_element(weights.begin(),weights.end());
	int sum = 0;
	for (int &i : weights) {
		sum += i;
	}
	int r = sum;
	while (l < r) {
		int mid = l + ((r - l) >> 1);
		if (!possible2(weights, D, mid)) {
			l = mid + 1;
		}
		else
			r = mid;
			
	}
	return l;
}



//27.移除元素
int leetcode::removeElement(vector<int>& nums, int val) //暴力解法，双层循环，发现目标 后面的元素往前进一位覆盖
{
	int size = nums.size();
	for (int i = 0; i < size; i++) {
		if (nums[i] == val) {//发现需要移除的元素，就将数组集体往前移动一位
			for (int j = i + 1; j < size; j++) {
				nums[j - 1] = nums[j];
			}
			i--;//因为下标i以后的数值都向前移动了一位，所以i也向前移动一位，之前的下一个元素已经移到i当前的位置，如果不i-1，就会跳过检查这个移过来的元素
			size--;//此时数值打大小-1
		}
	}
	return size;
}

int leetcode::removeElementSzz(vector<int>& nums, int val)
{
	int slowIndex = 0;
	for (int fastIndex = 0; fastIndex < nums.size(); ++fastIndex) {
		if (val != nums[fastIndex]) {
			nums[slowIndex++] = nums[fastIndex];
		}
	}
	return slowIndex;
}
//解法1：排序
/*首先将数组排序。

如果数组中全是非负数，则排序后最大的三个数相乘即为最大乘积；如果全是非正数，则最大的三个数相乘同样也为最大乘积。

如果数组中有正数有负数，则最大乘积既可能是三个最大正数的乘积，也可能是两个最小负数（即绝对值最大）与最大正数的乘积。

综上，我们在给数组排序后，分别求出三个最大正数的乘积，以及两个最小负数与最大正数的乘积，二者之间的最大值即为所求答案。

作者：LeetCode-Solution
链接：https://leetcode-cn.com/problems/maximum-product-of-three-numbers/solution/san-ge-shu-de-zui-da-cheng-ji-by-leetcod-t9sb/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
*/
int leetcode::maximumProduct(vector<int>& nums)
{
	sort(nums.begin(), nums.end());
	int n = nums.size();
	return max(nums[0] * nums[1] * nums[n - 1], nums[n - 3] * nums[n - 2] * nums[n - 1]);
}

int leetcode::maximumProduct2(vector<int>& nums)
{
	// 最小的和第二小的
	int min1 = INT_MAX, min2 = INT_MAX;
	// 最大的、第二大的和第三大的
	int max1 = INT_MIN, max2 = INT_MIN, max3 = INT_MIN;

	for (int x : nums) {
		if (x < min1) {
			min2 = min1;
			min1 = x;
		}
		else if (x < min2) {
			min2 = x;
		}

		if (x > max1) {
			max3 = max2;
			max2 = max1;
			max1 = x;
		}
		else if (x > max2) {
			max3 = max2;
			max2 = x;
		}
		else if (x > max3) {
			max3 = x;
		}
	}

	return max(min1 * min2 * max1, max1 * max2 * max3);
}




/*方法一：快速幂 + 递归
* 快速幂算法」的本质是分治算法。举个例子，如果我们要计算 x^{64}x 
64，我们可以按照：x \to x^2 \to x^4 \to x^8 \to x^{16} \to x^{32} \to x^{64}
x→x 2
 →x 
4
 →x 
8
 →x 
16
 →x 
32
 →x 
64
 

的顺序，从 xx 开始，每次直接把上一次的结果进行平方，计算 66 次就可以得到 x^{64}x 
64
  的值，而不需要对 xx 乘 6363 次 xx。

再举一个例子，如果我们要计算 x^{77}x 
77
 ，我们可以按照：

x \to x^2 \to x^4 \to x^9 \to x^{19} \to x^{38} \to x^{77}
x→x 
2
 →x 
4
 →x 
9
 →x 
19
 →x 
38
 →x 
77
 

的顺序，在 x \to x^2x→x 
2
 ，x^2 \to x^4x 
2
 →x 
4
 ，x^{19} \to x^{38}x 
19
 →x 
38
  这些步骤中，我们直接把上一次的结果进行平方，而在 x^4 \to x^9x 
4
 →x 
9
 ，x^9 \to x^{19}x 
9
 →x 
19
 ，x^{38} \to x^{77}x 
38
 →x 
77
  这些步骤中，我们把上一次的结果进行平方后，还要额外乘一个 xx。

直接从左到右进行推导看上去很困难，因为在每一步中，我们不知道在将上一次的结果平方之后，还需不需要额外乘 xx。但如果我们从右往左看，分治的思想就十分明显了：

当我们要计算 x^nx 
n
  时，我们可以先递归地计算出 y = x^{\lfloor n/2 \rfloor}y=x 
⌊n/2⌋
 ，其中 \lfloor a \rfloor⌊a⌋ 表示对 aa 进行下取整；

根据递归计算的结果，如果 nn 为偶数，那么 x^n = y^2x 
n
 =y 
2
 ；如果 nn 为奇数，那么 x^n = y^2 * xx 
n
 =y 
2
 ∗x；

递归的边界为 n = 0n=0，任意数的 00 次方均为 11。

由于每次递归都会使得指数减少一半，因此递归的层数为 O(\log n)O(logn)，算法可以在很快的时间内得到结果。

作者：LeetCode-Solution
链接：https://leetcode-cn.com/problems/powx-n/solution/powx-n-by-leetcode-solution/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
*/
double leetcode::myPow(double x, int n)
{
	long long N = n;
	return N >= 0?quickMul(x,N):1.0/quickMul(x,-N);
}

double leetcode::quickMul(double x, long long N)
{
	if (N == 0) {
		return 1.0;
	}
	double ans = quickMul(x, N / 2);//递归中 往上一层递归传递参数
	return N %2==0?ans*ans:ans*ans*x;//往下一层传递该终止递归层的返回值
}

double leetcode::quickMul2(double x, long long N)
{
	double ans = 1.0;
	//贡献的初始值为x
	double x_c = x;
	//对N进行二进制拆分的时候同时计算答案
	while (N > 0) {
		if (N % 2 == 1) {
			//如果N二进制表示的最低位为1，那么需要计入贡献
			ans *= x_c;
		}
		//将贡献不断地平方
		x_c *= x_c;
		//舍弃N二进制表示的最低位，这样我们每次只要判断最低位即可
		N /= 2;
	}
	return ans;
}

//public double myPow(double x, int n) {
//	// 迭代算法，利用二进制位
//	if (x == 0) { // 0 的任何次方都等于 0,1 的任何次方都等于 1
//		return x;
//	}
//
//	long power = n;    // 为了保证-n不溢出，先转换成long类型
//	if (n < 0) {         // 如果n小于0， 求1/x的-n次方
//		power *= -1;
//		x = 1 / x;
//	}
//	double weight = x;  // 权值初值为x, 即二进制位第1位的权值为x^1
//	double res = 1;
//	while (power != 0) {
//		// 如果当前二进制位为1， 让结果乘上这个二进制位上的权值, 
//		// 该位权值在上一轮迭代中已经计算出来了
//		if ((power & 1) == 1) {
//			res *= weight;
//		}
//		weight *= weight;   // 计算下一个二进制位的权值
//		power /= 2;
//	}
//	return res;
//}

vector<int> leetcode::twoSum(vector<int>& nums, int target)
{
	if (nums.empty()) return {};
	std::unordered_map<int, int>ans;
	for (int i = 0; i < nums.size(); i++) {
		auto iter = ans.find(target - nums[i]);//如果在map中找不到值，find函数返回的是end()，再次强调end()返回的是最末一个元素的下一位
		if (iter != ans.end()) {
			return{ iter->second,i};
		}
		//ans.insert(nums[i], i);//不能直接这样插入，虽然编译不报错但是运行会出错，不支持这样

		ans.insert(make_pair(nums[i], i));//还可以ans.insert(pari<int,int>(nums[i], i));
		//ans[nums[i]] = i;//因为是按数组的值来匹配的，所以用数组的值作为key
	}
	return {};
}

vector<vector<int>> leetcode::threeSum(vector<int>& nums)
{
	if (nums.empty() || nums.size() < 3) return {};
	vector<vector<int> > result;
	sort(nums.begin(), nums.end());//对数组排序，保证a<=b<=c 这样取了abc就不会再取bac（如果数组后面还有和a一样大的值）
	//找出a+b+c = 0；
	//a = nums[i],b=nums[j],c=-(a+b);
	for (int i = 0; i < nums.size(); ++i) {
		//排序之后如果第一个元素已经大于0
		if (nums[0] > 0) {
			continue;
		}
		if (i > 0 && nums[i] == nums[i - 1]) {//三元祖元素a去重，每一重元素相邻元素不能相同，不然可能会取到重复的三元组
			continue;
		}
		unordered_set<int> set;//在a和b确定的情况下保证c不重复
		for (int j = i + 1; j < nums.size(); j++) {
			//为什么是j> i +　２呢　如果是ｊ　＞　ｉ　＋　１，｛０，０，０｝，就返回不了正确的答案
			//如果三重循环 可以返回0，0，0，但是这里是两重循环，第一次执行完毕是往set存入一个值，还没有返回
			//所以要确保能对于{0，0，0}这种能循环到底部
			if (j > i + 2 && nums[j] == nums[j - 1] && nums[j - 1] == nums[j - 2]) continue;//b去重
			int c = 0 - (nums[i] + nums[j]);//c=0-(a+b),
			if (set.find(c) != set.end()) {
				result.push_back({nums[i],nums[j],c});
				set.erase(c);//c去重,如果不去掉c，如果极端情况数组里都是0，就会重复返回
			}
			else {
				//插入的是nums[j]的值。当set中没有值等于c=0-(a+b)时，存入b的值，第二重循环继续往后找，a不变
				//不知道数组中有没有c这个值，除非遍历，所以先把b存下来，反正b这层还要继续往后循环，如果找到新b 成立a+新b+c = 0
				//a+b新+c1 = 0;(这个c1是set中已有的，就是之前的b） 式子转换为 a+b新 + b = 0;对比之前的a + b + c = 0;说明新找的b就是往后循环里找到的c
				set.insert(nums[j]);//不能插入c.因为不能保证c这个值存在于数组中
				//往set里存入nums[j]就代表着定死一个j了，相当于第二重循环，然后在接下来的循环中，边迭代地将num[j_new]插入边找能配对的c
			}
		}	
	}
	return result;
}

vector<vector<int>> leetcode::threeSum2(vector<int>& nums)
{
	if (nums.empty() || nums.size() < 3) return {};
	vector<vector<int> > result;
	sort(nums.begin(), nums.end());
	
	int n = nums.size();
	for (int i = 0; i < n; ++i) {
		if (nums[0] > 0) return {};
		if (i > 0 && nums[i] == nums[i - 1])
			continue;
		int third = n - 1;//c对应的指针初始指向数组的最左边
		
		for (int second = i + 1; second < n; ++second) {
			if (second > i + 1 && nums[second] == nums[second - 1])
				continue;
			while (second < third && nums[i] + nums[second] + nums[third] > 0) {//b应该始终小于c，保证只取abc不会重复取acb，递减寻找符合的值
				--third;
			}
			// 如果指针重合，随着 b 后续的增加
				// 就不会有满足 a+b+c=0 并且 b<c 的 c 了，可以退出循环
			if (second == third) {
				break;
			}
			if (nums[i] + nums[second] + nums[third] == 0) {
				result.push_back({ nums[i], nums[second], nums[third] });
			}

		}
	}
	return result;
}

vector<vector<int>> leetcode::threeSum3(vector<int>& nums)
{
	int size = nums.size();
	if (size < 3)   return {};          // 特判
	vector<vector<int> >res;            // 保存结果（所有不重复的三元组）
	std::sort(nums.begin(), nums.end());// 排序（默认递增）
	for (int i = 0; i < size; i++)      // 固定第一个数，转化为求两数之和
	{
		if (nums[i] > 0)    return res; // 第一个数大于 0，后面都是递增正数，不可能相加为零了
		// 去重：如果此数已经选取过，跳过
		if (i > 0 && nums[i] == nums[i - 1])  continue;
		// 双指针在nums[i]后面的区间中寻找和为0-nums[i]的另外两个数
		int left = i + 1;
		int right = size - 1;
		while (left < right)
		{
			if (nums[left] + nums[right] > -nums[i])
				right--;    // 两数之和太大，右指针左移
			else if (nums[left] + nums[right] < -nums[i])
				left++;     // 两数之和太小，左指针右移
			else
			{
				// 找到一个和为零的三元组，添加到结果中，左右指针内缩，继续寻找
				res.push_back(vector<int>{nums[i], nums[left], nums[right]});
				left++;
				right--;
				// 去重：第二个数和第三个数也不重复选取
				// 例如：[-4,1,1,1,2,3,3,3], i=0, left=1, right=5
				while (left < right && nums[left] == nums[left - 1])  left++;
				while (left < right && nums[right] == nums[right + 1])    right--;
			}
		}
	}
	return res;
}

int leetcode::findLengthOfLCIS(vector<int>& nums)
{
	if (nums.empty()) return 0;
	int ans = 1;
	int n = nums.size();
	int start = 0;//初始的子序列开始下标
	for (int i = 0; i < n; i++) {
		if (i > 0 && nums[i] <= nums[i - 1]) {
			start = i; //当前的数值小于等于前一个时，中断，更新新的start为当前的i
		}
		ans = max(ans, i - start + 1);
	}
	return ans;
}

bool leetcode::increasingTriplet(vector<int>& nums)
{
	if (nums.empty() || nums.size() < 3) return false;
	int n = nums.size();
	int one = INT_MAX, two = INT_MAX;
	for (int three : nums) {
		if (three > two) return true;
		else if (three <= one) one = three;
		else two = three;
	}
	return false;
}

