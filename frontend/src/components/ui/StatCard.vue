<template>
  <div class="card p-6">
    <div class="flex items-start justify-between">
      <div>
        <p class="text-sm font-medium text-secondary-500">{{ label }}</p>
        <p class="text-2xl font-bold text-secondary-900 mt-1">
          {{ formattedValue }}
        </p>
        <div v-if="change !== undefined" class="flex items-center gap-1 mt-2">
          <svg
            :class="[
              'w-4 h-4',
              change >= 0 ? 'text-success-500' : 'text-danger-500 rotate-180'
            ]"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 10l7-7m0 0l7 7m-7-7v18" />
          </svg>
          <span :class="change >= 0 ? 'text-success-600' : 'text-danger-600'" class="text-sm font-medium">
            {{ Math.abs(change) }}%
          </span>
          <span class="text-sm text-secondary-500">vs 上月</span>
        </div>
      </div>
      <div :class="['w-12 h-12 rounded-lg flex items-center justify-center', iconBgClass]">
        <component :is="icon" :class="['w-6 h-6', iconClass]" />
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, h } from 'vue'

const props = defineProps({
  label: {
    type: String,
    required: true
  },
  value: {
    type: [Number, String],
    required: true
  },
  change: {
    type: Number,
    default: undefined
  },
  icon: {
    type: Object,
    default: null
  },
  color: {
    type: String,
    default: 'primary',
    validator: (v) => ['primary', 'success', 'warning', 'danger'].includes(v)
  },
  prefix: {
    type: String,
    default: ''
  },
  suffix: {
    type: String,
    default: ''
  }
})

const formattedValue = computed(() => {
  return `${props.prefix}${props.value}${props.suffix}`
})

const iconBgClass = computed(() => {
  const classes = {
    primary: 'bg-primary-100',
    success: 'bg-success-100',
    warning: 'bg-warning-100',
    danger: 'bg-danger-100'
  }
  return classes[props.color]
})

const iconClass = computed(() => {
  const classes = {
    primary: 'text-primary-600',
    success: 'text-success-600',
    warning: 'text-warning-600',
    danger: 'text-danger-600'
  }
  return classes[props.color]
})
</script>
