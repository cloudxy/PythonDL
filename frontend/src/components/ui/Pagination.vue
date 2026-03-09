<template>
  <div class="flex items-center gap-2">
    <span class="text-sm text-secondary-500">每页显示</span>
    <select
      :value="pageSize"
      @change="handlePageSizeChange"
      class="form-input py-1 text-sm w-20"
    >
      <option v-for="size in pageSizeOptions" :key="size" :value="size">{{ size }}</option>
    </select>
    <span class="text-sm text-secondary-500">条</span>
    <span class="text-sm text-secondary-500 ml-4">
      共 {{ total }} 条
    </span>
  </div>
</template>

<script setup>
const props = defineProps({
  pageSize: {
    type: Number,
    default: 10
  },
  total: {
    type: Number,
    default: 0
  },
  pageSizeOptions: {
    type: Array,
    default: () => [10, 20, 50, 100]
  }
})

const emit = defineEmits(['update:pageSize'])

const handlePageSizeChange = (event) => {
  emit('update:pageSize', Number(event.target.value))
}
</script>
