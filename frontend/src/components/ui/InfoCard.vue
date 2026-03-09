<template>
  <div class="card">
    <div class="p-4 border-b border-secondary-100">
      <div class="flex items-center justify-between">
        <h3 class="font-semibold text-secondary-900">{{ title }}</h3>
        <Badge :variant="statusVariant">{{ status }}</Badge>
      </div>
      <p class="text-sm text-secondary-500 mt-1">{{ description }}</p>
    </div>
    <div class="p-4 space-y-3">
      <div v-for="(item, index) in items" :key="index" class="flex items-center justify-between">
        <span class="text-sm text-secondary-600">{{ item.label }}</span>
        <span class="text-sm font-medium text-secondary-900">{{ item.value }}</span>
      </div>
    </div>
    <div v-if="$slots.footer" class="p-4 border-t border-secondary-100 bg-secondary-50">
      <slot name="footer" />
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import Badge from './Badge.vue'

const props = defineProps({
  title: {
    type: String,
    required: true
  },
  description: {
    type: String,
    default: ''
  },
  status: {
    type: String,
    default: ''
  },
  statusVariant: {
    type: String,
    default: 'primary'
  },
  items: {
    type: Array,
    default: () => []
  }
})
</script>
