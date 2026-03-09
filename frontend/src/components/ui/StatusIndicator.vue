<template>
  <div class="flex items-center gap-2">
    <div class="w-2 h-2 rounded-full" :class="statusClass"></div>
    <span class="text-sm" :class="textClass">{{ text }}</span>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  status: {
    type: String,
    default: 'default',
    validator: (v) => ['default', 'success', 'warning', 'danger', 'info'].includes(v)
  },
  text: {
    type: String,
    default: ''
  }
})

const statusClass = computed(() => {
  const classes = {
    default: 'bg-secondary-400',
    success: 'bg-success-500',
    warning: 'bg-warning-500',
    danger: 'bg-danger-500',
    info: 'bg-primary-500'
  }
  return classes[props.status]
})

const textClass = computed(() => {
  const classes = {
    default: 'text-secondary-600',
    success: 'text-success-600',
    warning: 'text-warning-600',
    danger: 'text-danger-600',
    info: 'text-primary-600'
  }
  return classes[props.status]
})
</script>
