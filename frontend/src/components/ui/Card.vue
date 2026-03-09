<template>
  <div class="card">
    <div v-if="$slots.header || title" class="card-header flex items-center justify-between">
      <div>
        <h3 v-if="title" class="text-lg font-semibold text-secondary-900">{{ title }}</h3>
        <p v-if="subtitle" class="text-sm text-secondary-500 mt-1">{{ subtitle }}</p>
        <slot name="header" />
      </div>
      <div v-if="$slots.actions" class="flex items-center gap-2">
        <slot name="actions" />
      </div>
    </div>
    <div class="card-body">
      <slot />
    </div>
    <div v-if="$slots.footer" class="card-footer">
      <slot name="footer" />
    </div>
  </div>
</template>

<script setup>
defineProps({
  title: {
    type: String,
    default: ''
  },
  subtitle: {
    type: String,
    default: ''
  }
})
</script>
