public class Practice {

    public void mergeSort(int[] nums) {
        int mid = nums.length / 2;
        
        int[] left = Array.copy(nums, 0, mid);
        int[] right = Array.copy(nums, mid, nums.length);

        mergeSort(left);
        mergeSort(right);

        merge(nums, left, right);
    }

    private static void merge(int[] arr, int[] left, int[] right) {
        int i = 0, j =0, k = 0;
        while(i < left.length && j < right.length) {
            if(left[i] < right[j]) {
                arr[k] = left[i];
                i++;
            } else {
                arr[k] = right[j];
                j++;
            }
        }

        while(i < left.length) {
            arr[k] = left[i];
            k++;
            i++;
        }

        while(j < right.length) {
            arr[k] = right[j];
            k++;
            j++;
        }
    }

}