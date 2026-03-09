<template>
  <div class="card overflow-hidden">
    <div class="px-6 py-4 border-b border-secondary-100">
      <h3 class="text-lg font-semibold text-secondary-900">{{ title }}</h3>
    </div>
    <div class="p-6">
      <slot />
    </div>
  </div>
</template>

<script setup>
defineProps({
  title: {
    type: String,
    required: true
  }
})
</script>
