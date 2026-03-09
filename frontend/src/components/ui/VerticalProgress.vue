<template>
  <div class="relative">
    <div class="h-32 bg-secondary-100 rounded-lg overflow-hidden">
      <div
        class="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-primary-500 to-primary-400 transition-all duration-300"
        :style="{ height: `${percentage}%` }"
      ></div>
    </div>
    <div class="mt-2 text-center">
      <span class="text-sm font-medium text-secondary-900">{{ value }}</span>
      <span class="text-xs text-secondary-500"> / {{ max }}</span>
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
  }
})

const percentage = computed(() => {
  return Math.min(Math.max((props.value / props.max) * 100, 0), 100)
})
</script>
