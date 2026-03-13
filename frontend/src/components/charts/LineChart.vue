<template>
  <div class="w-full h-full">
    <Line v-if="loaded" :data="chartData" :options="chartOptions" />
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { Line } from 'vue-chartjs'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

const props = defineProps({
  title: {
    type: String,
    default: ''
  },
  labels: {
    type: Array,
    required: true
  },
  datasets: {
    type: Array,
    required: true
  },
  showLegend: {
    type: Boolean,
    default: true
  },
  showTooltip: {
    type: Boolean,
    default: true
  },
  fill: {
    type: Boolean,
    default: false
  },
  smooth: {
    type: Boolean,
    default: true
  },
  height: {
    type: Number,
    default: 300
  }
})

const loaded = ref(false)

onMounted(() => {
  loaded.value = true
})

const chartData = computed(() => ({
  labels: props.labels,
  datasets: props.datasets.map((dataset, index) => ({
    ...dataset,
    fill: props.fill,
    tension: props.smooth ? 0.4 : 0,
    borderColor: dataset.borderColor || getColor(index),
    backgroundColor: dataset.backgroundColor || getColor(index, 0.2),
    pointRadius: dataset.pointRadius ?? 3,
    pointHoverRadius: dataset.pointHoverRadius ?? 5
  }))
}))

const chartOptions = computed(() => ({
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    title: {
      display: !!props.title,
      text: props.title,
      font: {
        size: 16,
        weight: 'bold'
      },
      padding: 20
    },
    legend: {
      display: props.showLegend,
      position: 'top',
      labels: {
        usePointStyle: true,
        pointStyle: 'circle'
      }
    },
    tooltip: {
      enabled: props.showTooltip,
      mode: 'index',
      intersect: false,
      backgroundColor: 'rgba(0, 0, 0, 0.8)',
      titleFont: {
        size: 14
      },
      bodyFont: {
        size: 13
      },
      padding: 12,
      cornerRadius: 8,
      displayColors: true
    }
  },
  scales: {
    x: {
      grid: {
        display: false
      },
      ticks: {
        maxRotation: 45,
        minRotation: 45
      }
    },
    y: {
      grid: {
        color: 'rgba(0, 0, 0, 0.05)'
      },
      beginAtZero: false
    }
  },
  interaction: {
    mode: 'nearest',
    axis: 'x',
    intersect: false
  }
}))

function getColor(index, alpha = 1) {
  const colors = [
    `rgba(59, 130, 246, ${alpha})`,
    `rgba(16, 185, 129, ${alpha})`,
    `rgba(245, 158, 11, ${alpha})`,
    `rgba(239, 68, 68, ${alpha})`,
    `rgba(139, 92, 246, ${alpha})`,
    `rgba(236, 72, 153, ${alpha})`,
    `rgba(14, 165, 233, ${alpha})`,
    `rgba(34, 197, 94, ${alpha})`
  ]
  return colors[index % colors.length]
}
</script>

<style scoped>
div {
  height: v-bind('height + "px"');
}
</style>
