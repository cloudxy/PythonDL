<template>
  <div class="space-y-6">
    <div v-if="title || description" class="pb-4 border-b border-secondary-200">
      <h3 v-if="title" class="text-lg font-semibold text-secondary-900">{{ title }}</h3>
      <p v-if="description" class="text-sm text-secondary-500 mt-1">{{ description }}</p>
    </div>
    <div :class="gridClass">
      <slot />
    </div>
    <div v-if="$slots.actions" class="flex items-center justify-end gap-3 pt-4 border-t border-secondary-200">
      <slot name="actions" />
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  title: {
    type: String,
    default: ''
  },
  description: {
    type: String,
    default: ''
  },
  columns: {
    type: Number,
    default: 1,
    validator: (v) => [1, 2, 3].includes(v)
  }
})

const gridClass = computed(() => {
  const classes = {
    1: 'space-y-4',
    2: 'grid grid-cols-1 md:grid-cols-2 gap-4',
    3: 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4'
  }
  return classes[props.columns]
})
</script>
