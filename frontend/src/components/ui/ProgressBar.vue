<template>
  <div class="relative">
    <div class="flex items-center justify-between mb-2">
      <span class="text-sm font-medium text-secondary-700">{{ label }}</span>
      <span class="text-sm text-secondary-500">{{ value }} / {{ max }}</span>
    </div>
    <div class="h-2 bg-secondary-200 rounded-full overflow-hidden">
      <div
        class="h-full rounded-full transition-all duration-300"
        :class="colorClass"
        :style="{ width: `${percentage}%` }"
      ></div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  label: {
    type: String,
    default: ''
  },
  value: {
    type: Number,
    default: 0
  },
  max: {
    type: Number,
    default: 100
  },
  color: {
    type: String,
    default: 'primary',
    validator: (v) => ['primary', 'success', 'warning', 'danger'].includes(v)
  }
})

const percentage = computed(() => {
  return Math.min(Math.max((props.value / props.max) * 100, 0), 100)
})

const colorClass = computed(() => {
  const classes = {
    primary: 'bg-primary-500',
    success: 'bg-success-500',
    warning: 'bg-warning-500',
    danger: 'bg-danger-500'
  }
  return classes[props.color]
})
</script>
