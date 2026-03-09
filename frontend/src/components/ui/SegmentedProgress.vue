<template>
  <div class="relative">
    <div class="flex items-center gap-4">
      <div class="flex-1">
        <div class="h-2 bg-secondary-200 rounded-full overflow-hidden">
          <div
            v-for="(segment, index) in segments"
            :key="index"
            class="h-full float-left transition-all duration-300"
            :class="segment.colorClass"
            :style="{ width: `${segment.percentage}%` }"
          ></div>
        </div>
      </div>
    </div>
    <div v-if="showLegend" class="flex flex-wrap gap-4 mt-3">
      <div v-for="(segment, index) in segments" :key="index" class="flex items-center gap-2">
        <div class="w-3 h-3 rounded-full" :class="segment.colorClass"></div>
        <span class="text-sm text-secondary-600">{{ segment.label }}</span>
        <span class="text-sm font-medium text-secondary-900">{{ segment.value }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  data: {
    type: Array,
    required: true
  },
  showLegend: {
    type: Boolean,
    default: true
  }
})

const segments = computed(() => {
  const total = props.data.reduce((sum, item) => sum + item.value, 0)
  const colorClasses = {
    primary: 'bg-primary-500',
    success: 'bg-success-500',
    warning: 'bg-warning-500',
    danger: 'bg-danger-500',
    secondary: 'bg-secondary-500'
  }
  
  return props.data.map(item => ({
    ...item,
    percentage: total > 0 ? (item.value / total) * 100 : 0,
    colorClass: colorClasses[item.color] || 'bg-secondary-500'
  }))
})
</script>
