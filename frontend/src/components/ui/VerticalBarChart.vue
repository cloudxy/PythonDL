<template>
  <div class="card p-6">
    <div class="flex items-center justify-between mb-4">
      <h3 class="font-semibold text-secondary-900">{{ title }}</h3>
      <slot name="actions" />
    </div>
    <div class="flex items-end gap-2 h-40">
      <div
        v-for="(item, index) in items"
        :key="index"
        class="flex-1 flex flex-col items-center"
      >
        <div
          :class="[
            'w-full rounded-t transition-all duration-300',
            colorClass(item.color)
          ]"
          :style="{ height: `${(item.value / maxValue) * 100}%` }"
        ></div>
        <span class="text-xs text-secondary-500 mt-2">{{ item.label }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  title: {
    type: String,
    required: true
  },
  items: {
    type: Array,
    default: () => []
  }
})

const maxValue = computed(() => {
  return Math.max(...props.items.map(item => item.value), 1)
})

const colorClass = (color = 'primary') => {
  const classes = {
    primary: 'bg-primary-500',
    success: 'bg-success-500',
    warning: 'bg-warning-500',
    danger: 'bg-danger-500',
    secondary: 'bg-secondary-500'
  }
  return classes[color] || 'bg-primary-500'
}
</script>
