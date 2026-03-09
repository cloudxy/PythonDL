<template>
  <div class="relative">
    <div class="h-2 bg-secondary-200 rounded-full overflow-hidden">
      <div
        :class="progressClass"
        class="h-full rounded-full transition-all duration-300"
        :style="{ width: `${percentage}%` }"
      ></div>
    </div>
    <div v-if="showLabel" class="flex justify-between mt-1">
      <span class="text-xs text-secondary-500">{{ label }}</span>
      <span class="text-xs font-medium text-secondary-700">{{ percentage }}%</span>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
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
  },
  showLabel: {
    type: Boolean,
    default: false
  },
  label: {
    type: String,
    default: ''
  }
})

const percentage = computed(() => {
  return Math.min(Math.max((props.value / props.max) * 100, 0), 100)
})

const progressClass = computed(() => {
  const classes = {
    primary: 'bg-primary-500',
    success: 'bg-success-500',
    warning: 'bg-warning-500',
    danger: 'bg-danger-500'
  }
  return classes[props.color]
})
</script>
