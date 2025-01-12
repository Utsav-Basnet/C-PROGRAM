class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        // Initialize a dummy node to simplify the process of adding nodes
        dummy = ListNode()
        current = dummy  // This will point to the current node in the result list
        carry = 0
        
        // Traverse both lists and add corresponding digits along with carry
        while l1 or l2 or carry:
            // Get the current values, or 0 if the list has ended
            val1 = l1.val if l1 else 0
            val2 = l2.val if l2 else 0
            
            // Sum the values and the carry
            total = val1 + val2 + carry
            
            // Update carry (either 0 or 1)
            carry = total // 10
            
            // Create a new node with the current digit (total % 10)
            current.next = ListNode(total % 10)
            current = current.next  // Move to the next node
            
            // Move to the next nodes in the input lists if available
            if l1: l1 = l1.next








class Solution:
    def findMedianSortedArrays(self, nums1, nums2):
        // Merge the two sorted arrays
        merged = sorted(nums1 + nums2)
        
        // Find the median
        length = len(merged)
        if length % 2 == 1:
            return merged[length // 2]
        else:
            return (merged[length // 2 - 1] + merged[length // 2]) / 2








class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        // Dictionary to store the last seen index of characters
        char_map = {}
        left = 0  // left pointer
        max_length = 0  // to store the length of the longest substring

        // Traverse the string with right pointer
        for right in range(len(s)):
            // If the character is already in the window, move the left pointer
            if s[right] in char_map and char_map[s[right]] >= left:
                left = char_map[s[right]] + 1  // move left pointer past the previous occurrence

            // Update or add the current character's index in the map
            char_map[s[right]] = right

            // Calculate the length of the current window
            max_length = max(max_length, right - left + 1)

        return max_length  // Returning the result as an integer

// Example usage:
solution = Solution()

// Test case 1
s1 = "abcabcbb"
print(solution.lengthOfLongestSubstring(s1))  // Expected Output: 3

// Test case 2
s2 = "bbbbb"
print(solution.lengthOfLongestSubstring(s2))  // Expected Output: 1

// Test case 3
s3 = "pwwkew"
print(solution.lengthOfLongestSubstring(s3))  // Expected Output: 3














class Solution:
    def isPalindrome(self, x: int) -> bool:
        // Negative numbers and numbers ending in 0 (except 0 itself) are not palindromes
        if x < 0 or (x % 10 == 0 and x != 0):
            return False
        
        reversed_half = 0
        while x > reversed_half:
            // Pop the last digit from x and append it to reversed_half
            reversed_half = reversed_half * 10 + x % 10
            x //= 10
        
        // Check if the number is a palindrome by comparing x with reversed_half
        // For odd digit numbers, reversed_half // 10 removes the middle digit
        return x == reversed_half or x == reversed_half // 10

// Example usa










class Solution:
    def romanToInt(self, s: str) -> int:
        // Mapping of Roman numeral symbols to integer values
        roman_map = {
            'I': 1,
            'V': 5,
            'X': 10,
            'L': 50,
            'C': 100,
            'D': 500,
            'M': 1000
        }
        
        total = 0
        for i in range(len(s)):
            // Get the value of the current Roman numeral
            current_value = roman_map[s[i]]
            
            // If this value is less than the next value, subtract it
            if i + 1 < len(s) and current_value < roman_map[s[i + 1]]:
                total -= current_value
            else:
                // This is the 'else' block. Ensure it is indented properly
                total += current_value 
        
        return total

// Example usage
solution = Solution()

// Test case 1
s1 = "III"
print(solution.romanToInt(s1))  // Expected Output: 3

// Test case 2
s2 = "LVIII"
print(solution.romanToInt(s2))  // Expected Output: 58

// Test case 3
s3 = "MCMXCIV"
print(solution.romanToInt(s3))  // Expected Output: 1994







class Solution:
    def longestCommonPrefix(self, strs):
        // Edge case: if the list is empty, return an empty string
        if not strs:
            return ""

        // Start with the first string as the initial prefix
        prefix = strs[0]
        
        // Compare the prefix with each string in the list
        for string in strs[1:]:
            // Check the common prefix between the current string and the prefix
            while not string.startswith(prefix):
                // If the current string doesn't start with the prefix, shorten the prefix
                prefix = prefix[:-1]
                // If there's no common prefix, return an empty string
                if not prefix:
                    return ""
        
        return prefix

// Example usage
solution = Solution()










class Solution:
    def isValid(self, s: str) -> bool:
        // Stack to store opening brackets
        stack = []
        
        // HashMap to match closing brackets with opening brackets
        bracket_map = {')': '(', '}': '{', ']': '['}
        
        // Iterate through the string
        for char in s:
            if char in bracket_map.values():
                // If it's an opening bracket, push it to the stack
                stack.append(char)
            elif char in bracket_map:
                // If it's a closing bracket, check the stack
                if stack and stack[-1] == bracket_map[char]:
                    stack.pop()  // If it matches the top of the stack, pop it
                else:
                    return False  // If it doesn't match, return False
        
        // If the stack is empty, all brackets matched correctly
        return not stack











class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def mergeTwoLists(self, list1: ListNode, list2: ListNode) -> ListNode:
        // Create a dummy node to start the merged list
        dummy = ListNode()
        current = dummy
        
        // Traverse through both lists
        while list1 and list2:
            if list1.val < list2.val:
                current.next = list1
                list1 = list1.next
            else:
                current.next = list2
                list2 = list2.next
            current = current.next
        
        // If any elements are left in list1 or list2, attach the remaining part
        if list1:
            current.next = list1
        if list2:
            current.next = list2
        
        // Return the merged list, starting from the node after the dummy node
        return dummy.next






class Solution:
    def removeDuplicates(self, nums):
        // Check if the array is empty
        if not nums:
            return 0
        
        // Pointer for the position of the last unique element
        i = 0
        
        // Iterate through the array starting from the second element
        for j in range(1, len(nums)):
            // If the current element is not equal to the last unique element
            if nums[j] != nums[i]:
                // Increment i and update the value at nums[i]
                i += 1
                nums[i] = nums[j]
        
        // The number of unique elements is i + 1
        return i + 1







class Solution:
    def removeElement(self, nums, val):
        // Initialize a pointer for the position of the next valid element
        k = 0
        
        // Iterate through the array
        for i in range(len(nums)):
            if nums[i] != val:
                // Place the current element at position k and increment k
                nums[k] = nums[i]
                k += 1
        
        // Return the number of elements that are not equal to val
        return k







class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        // Use the find() method to get the first occurrence index
        return haystack.find(needle)





class Solution:
    def searchInsert(self, nums, target):
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return left  // left is the insertion point if target is not found






class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        // Strip leading/trailing spaces and split the string by spaces
        words = s.strip().split()
        
        // The last word is the last element of the list
        return len(words[-1])





class Solution:
    def plusOne(self, digits):
        // Start from the last digit and process the carry
        n = len(digits)
        
        for i in range(n - 1, -1, -1):
            if digits[i] < 9:
                digits[i] += 1
                return digits
            digits[i] = 0  // reset the current digit to 0 if it's 9
        
        // If we finished the loop, it means we have a carry to add
        return [1] + digits  // Add 1 at the beginning for the carry



class Solution:
    def addBinary(self, a: str, b: str) -> str:
        // Initialize pointers for both strings and the carry value
        i, j, carry = len(a) - 1, len(b) - 1, 0
        result = []

        // Loop through both strings from the end to the beginning
        while i >= 0 or j >= 0 or carry:
            // Get the current bits from both strings, default to 0 if the index is out of bounds
            bit_a = int(a[i]) if i >= 0 else 0
            bit_b = int(b[j]) if j >= 0 else 0

            // Sum the bits and the carry
            total = bit_a + bit_b + carry
            result.append(str(total % 2))  // Append the result of total % 2 (either 0 or 1)
            carry = total // 2  // Update carry (either 0 or 1)

            // Move to the next bits in both strings
            i -= 1
            j -= 1

        // The result list is in reverse order, so reverse it and join to form the final string
        return ''.join(result[::-1])






class Solution:
    def mySqrt(self, x: int) -> int:
        if x == 0:
            return 0
        
        left, right = 0, x
        
        while left <= right:
            mid = (left + right) // 2
            if mid * mid == x:
                return mid
            elif mid * mid < x:
                left = mid + 1
            else:
                right = mid - 1
        
        return right  // The right pointer will be the floor of the square root
        



class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 1:
            return 1
        
        // Initialize the first two steps
        prev2, prev1 = 1, 2
        
        for i in range(3, n + 1):
            // The number of ways to reach the current step is the sum of the previous two steps
            current = prev1 + prev2
            prev2 = prev1
            prev1 = current
        
        return prev1




class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        // Pointer to iterate through the list
        current = head
        
        // Traverse the list until the end
        while current and current.next:
            // If current value is equal to the next value, skip the next node
            if current.val == current.next.val:
                current.next = current.next.next
            else:
                current = current.next
        
        // Return the head of the modified list
        return head




class Solution:
    def merge(self, nums1, m, nums2, n):
        // Pointers for nums1, nums2, and the last position of nums1
        i, j, k = m - 1, n - 1, m + n - 1

        // Merge in reverse order
        while i >= 0 and j >= 0:
            if nums1[i] > nums2[j]:
                nums1[k] = nums1[i]
                i -= 1
            else:
                nums1[k] = nums2[j]
                j -= 1
            k -= 1

        // If any elements are left in nums2, copy them
        while j >= 0:
            nums1[k] = nums2[j]
            j -= 1
            k -= 1




// Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def inorderTraversal(self, root: TreeNode):
        // Helper function to perform recursive inorder traversal
        result = []
        
        def inorder(node):
            if node:
                inorder(node.left)  // Traverse left subtree
                result.append(node.val)  // Visit the root
                inorder(node.right)  // Traverse right subtree
        
        inorder(root)  // Start the traversal from the root
        return result







// Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        // If both nodes are None, the trees are the same
        if not p and not q:
            return True
        
        // If one node is None and the other is not, the trees are different
        if not p or not q:
            return False
        
        // If the values of the current nodes are different, return False
        if p.val != q.val:
            return False
        
        // Recursively check the left and right subtrees
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

        



// Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        // Helper function to check if two trees are mirror images
        def isMirror(t1, t2):
            // Both are None, symmetric
            if not t1 and not t2:
                return True
            // One is None, the other is not, not symmetric
            if not t1 or not t2:
                return False
            // Check if the current nodes are equal and recursively check the mirror properties
            return (t1.val == t2.val) and isMirror(t1.left, t2.right) and isMirror(t1.right, t2.left)
        
        // Start the check from the root
        if not root:
            return True  // An empty tree is symmetric
        
        return isMirror(root.left, root.right)




// Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        // Base case: if the node is None, the depth is 0
        if not root:
            return 0
        
        // Recursive case: calculate the depth of the left and right subtrees
        left_depth = self.maxDepth(root.left)
        right_depth = self.maxDepth(root.right)
        
        // The depth of the current node is 1 + the maximum depth of the subtrees
        return 1 + max(left_depth, right_depth)







// Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        
        // Helper function to check the height and balance of the tree
        def check_balance(node):
            if not node:
                return 0  // Height of an empty node is 0
            
            // Recursively get the height of the left and right subtrees
            left_height = check_balance(node.left)
            if left_height == -1:  // Left subtree is unbalanced
                return -1
            
            right_height = check_balance(node.right)
            if right_height == -1:  // Right subtree is unbalanced
                return -1
            
            // If the current node's left and right subtrees differ in height by more than 1
            if abs(left_height - right_height) > 1:
                return -1  // Return -1 to indicate imbalance
            
            // Return the height of the current node (max of left and right heights + 1)
            return max(left_height, right_height) + 1
        
        // If the tree is balanced, the result will not be -1
        return check_balance(root) != -1

// Example usage:
solution = Solution()

// Example 1: Balanced tree
root1 = TreeNode(3)
root1.left = TreeNode(9)
root1.right = TreeNode(20)
root1.right.left = TreeNode(15)
root1.right.right = TreeNode(7)
print(solution.isBalanced(root1))  // Expected: True

// Example 2: Unbalanced tree
root2 = TreeNode(1)
root2.left = TreeNode(2)
root2.right = TreeNode(2)
root2.left.left = TreeNode(3)
root2.left.right = TreeNode(3)
root2.left.left.left = TreeNode(4)
root2.left.left.right = TreeNode(4)
print(solution.isBalanced(root2))  // Expected: False

// Example 3: Empty tree
root3 = None
print(solution.isBalanced(root3))  // Expected: True











from collections import deque

// Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def minDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        // Initialize a queue for BFS
        queue = deque([(root, 1)])  // Queue will store pairs (node, depth)
        
        while queue:
            node, depth = queue.popleft()
            
            // If the current node is a leaf node, return the depth
            if not node.left and not node.right:
                return depth
            
            // Otherwise, add the children to the queue with an incremented depth
            if node.left:
                queue.append((node.left, depth + 1))
            if node.right:
                queue.append((node.right, depth + 1))
                
// Example usage:
solution = Solution()

// Example 1:
root1 = TreeNode(3)
root1.left = TreeNode(9)
root1.right = TreeNode(20)
root1.right.left = TreeNode(15)
root1.right.right = TreeNode(7)
print(solution.minDepth(root1))  // Output: 2

// Example 2:
root2 = TreeNode(2)
root2.right = TreeNode(3)
root2.right.right = TreeNode(4)
root2.right.right.right = TreeNode(5)
root2.right.right.right.right = TreeNode(6)
print(solution.minDepth(root2))  // Output: 5
