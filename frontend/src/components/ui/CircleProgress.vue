<template>
  <div class="relative">
    <svg class="w-full h-full" viewBox="0 0 36 36">
      <path
        class="text-secondary-200"
        stroke="currentColor"
        stroke-width="3"
        fill="none"
        :d="circlePath"
      />
      <path
        :class="colorClass"
        stroke="currentColor"
        stroke-width="3"
        fill="none"
        :stroke-dasharray="circumference"
        :stroke-dashoffset="offset"
        stroke-linecap="round"
        :d="circlePath"
        class="transition-all duration-300"
      />
    </svg>
    <div class="absolute inset-0 flex items-center justify-center">
      <span class="text-2xl font-bold text-secondary-900">{{ percentage }}%</span>
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
  }
})

const percentage = computed(() => {
  return Math.round(Math.min(Math.max((props.value / props.max) * 100, 0), 100))
})

const radius = 16
const circumference = 2 * Math.PI * radius

const circlePath = computed(() => {
  return `M 18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831`
})

const offset = computed(() => {
  return circumference - (percentage.value / 100) * circumference
})

const colorClass = computed(() => {
  const classes = {
    primary: 'text-primary-500',
    success: 'text-success-500',
    warning: 'text-warning-500',
    danger: 'text-danger-500'
  }
  return classes[props.color]
})
</script>
