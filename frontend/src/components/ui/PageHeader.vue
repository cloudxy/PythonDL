<template>
  <div class="flex items-center justify-between mb-6">
    <div>
      <h1 class="text-2xl font-bold text-secondary-900">{{ title }}</h1>
      <p v-if="subtitle" class="text-secondary-500 mt-1">{{ subtitle }}</p>
    </div>
    <div v-if="$slots.actions" class="flex items-center gap-3">
      <slot name="actions" />
    </div>
  </div>
</template>

<script setup>
defineProps({
  title: {
    type: String,
    required: true
  },
  subtitle: {
    type: String,
    default: ''
  }
})
</script>
