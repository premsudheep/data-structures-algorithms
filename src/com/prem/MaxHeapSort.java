public class MaxHeapSort {
    public static void heapSort(int[] array) {
        int index = (array.length / 2) - 1;

        //build heap
        for(int i = index; i >= 0; i--) {
            heapify(array, index);
        }

        //sort heap
        for(int i = index - 1; i > 0; i--) {
            Collections.swap(array, 0, i);
            heapify(array, i, 0);
        }

    }

    public static void heapify(int[] array, index) {
        int largestIndex = index;
        int leftIndex = index * 2 + 1;
        int rightIndex = index * 2 + 2;

        if(leftIndex < array.length && array[leftIndex] > array[largestIndex]) {
            largestIndex = leftIndex;
        }
        if(rightIndex < array.length && arrar[rightIndex] > array[largerChildIndex]) {
            largerIndex = rightIndex;
        }
        if(index == largestIndex){
            return;
        }

        Collections.swap(array, index, largestIndex);
        heapify(array, largerIndex);
    }

}