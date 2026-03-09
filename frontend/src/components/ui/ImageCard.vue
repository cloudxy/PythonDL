<template>
  <div class="card overflow-hidden">
    <img
      v-if="image"
      :src="image"
      :alt="title"
      class="w-full h-48 object-cover"
    />
    <div class="p-4">
      <div class="flex items-start justify-between">
        <div>
          <h3 class="font-semibold text-secondary-900">{{ title }}</h3>
          <p v-if="subtitle" class="text-sm text-secondary-500 mt-1">{{ subtitle }}</p>
        </div>
        <Badge v-if="status" :variant="statusVariant">{{ status }}</Badge>
      </div>
      <p v-if="description" class="text-sm text-secondary-600 mt-3 line-clamp-2">{{ description }}</p>
      <div v-if="$slots.footer" class="mt-4 pt-4 border-t border-secondary-100">
        <slot name="footer" />
      </div>
      <div v-else-if="showDefaultFooter" class="mt-4 pt-4 border-t border-secondary-100 flex items-center justify-between">
        <div class="flex items-center gap-2">
          <Avatar v-if="avatar" :src="avatar" :name="author" size="sm" />
          <span class="text-sm text-secondary-600">{{ author }}</span>
        </div>
        <span class="text-sm text-secondary-500">{{ date }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import Badge from './Badge.vue'
import Avatar from './Avatar.vue'

const props = defineProps({
  title: {
    type: String,
    required: true
  },
  subtitle: {
    type: String,
    default: ''
  },
  description: {
    type: String,
    default: ''
  },
  image: {
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
  author: {
    type: String,
    default: ''
  },
  avatar: {
    type: String,
    default: ''
  },
  date: {
    type: String,
    default: ''
  },
  showDefaultFooter: {
    type: Boolean,
    default: false
  }
})
</script>

<style scoped>
.line-clamp-2 {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
</style>
