<template>
  <div class="flex items-center gap-2 text-sm text-secondary-500 mb-4">
    <router-link to="/" class="hover:text-primary-600 transition-colors">
      <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
      </svg>
    </router-link>
    <template v-for="(item, index) in items" :key="index">
      <svg class="w-4 h-4 text-secondary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
      </svg>
      <router-link
        v-if="item.path"
        :to="item.path"
        class="hover:text-primary-600 transition-colors"
      >
        {{ item.label }}
      </router-link>
      <span v-else class="text-secondary-900 font-medium">{{ item.label }}</span>
    </template>
  </div>
</template>

<script setup>
defineProps({
  items: {
    type: Array,
    required: true
  }
})
</script>
